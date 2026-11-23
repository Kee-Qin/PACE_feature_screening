"""Column encoder implementations for representation ablations."""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
def _randomize_module_params(module, seed=42):
    """Re-initialize all parameters of a module using Kaiming uniform init.

    LayerNorm weight (gamma) is initialized to 1 (not 0) because LN(x) = 0*(x-mu)/sigma + bias
    would zero out the entire output and turn the block into an identity via residual connections.
    Other 1D params (biases) are initialized to 0 as usual.
    """
    torch.manual_seed(seed)
    for name, param in module.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.kaiming_uniform_(param, a=5**0.5)
        elif 'weight' in name and param.dim() == 1:
            # LayerNorm weight (gamma): must be non-zero to avoid zeroing LN output
            torch.nn.init.ones_(param)
        elif param.dim() == 1:
            torch.nn.init.zeros_(param)


class TabICLProjectionEmbedder:

    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.col_embedder = None
        self.embed_dim = None
        self.n_cls_tokens = 4
        self.feature_group_size = 1
        self.loaded = False

    def load_model(self, model_path: str, seed: int = 42, device: str = None) -> bool:
        if self.loaded:
            return True
        if device is not None:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        try:
            from tabicl import TabICLClassifier
            model = TabICLClassifier(
                model_path=model_path, n_estimators=1,
                random_state=seed, device=str(self.device)
            )
            model.fit([[0, 1], [1, 0]], [0, 1])
            self.col_embedder = model.model_.col_embedder
            self.embed_dim = self.col_embedder.embed_dim
            self.n_cls_tokens = getattr(self.col_embedder, 'reserve_cls_tokens', 4)
            # v1.1: feature_group=False, in_linear(1 → 128); v2: feature_group="same", in_linear(3 → 128)
            col_fg = getattr(self.col_embedder, 'feature_group', False)
            if col_fg and col_fg is not False:
                self.feature_group_size = self.col_embedder.feature_group_size
            else:
                self.feature_group_size = 1
            self.loaded = True
            print(f"TabICLProjectionEmbedder loaded: embed_dim={self.embed_dim}, "
                  f"group_size={self.feature_group_size}", flush=True)
            return True
        except Exception as e:
            print(f"TabICLProjectionEmbedder failed: {e}", flush=True)
            return False

    def get_projection_embeddings(self, X, batch_size: int = 32) -> torch.Tensor:
        import scipy.stats as sp_stats

        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)

        n, p = X_np.shape

        # Rank-based Gaussian transform
        gaussian_quantiles = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        X_processed = np.zeros((n, p), dtype=np.float32)
        for j in range(p):
            ranks = sp_stats.rankdata(X_np[:, j], method='average')
            X_processed[:, j] = gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]

        results = torch.zeros(n, p, self.embed_dim, device=self.device)
        n_cls = self.n_cls_tokens
        fpg = self.feature_group_size

        self.col_embedder.eval()
        with torch.no_grad():
            for start in range(0, p, batch_size):
                end = min(start + batch_size, p)
                chunk_p = end - start


                x_input = np.zeros((chunk_p, 1 + n_cls, n, fpg), dtype=np.float32)
                if fpg == 1:
                    x_input[:, n_cls, :, 0] = X_processed[:, start:end].T
                else:

                    for i in range(fpg):
                        x_input[:, n_cls, :, i] = X_processed[:, start:end].T

                x_tensor = torch.as_tensor(x_input, device=self.device)

                out = self.col_embedder.in_linear(x_tensor)
                # out shape: (chunk_p, 1+n_cls, n, embed_dim)
                emb = out[:, n_cls:, :, :].squeeze(1).permute(1, 0, 2)
                results[:, start:end, :] = emb.detach()

        return results

    def get_single_layer_embeddings(self, X, layers, batch_size=32):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")

        import scipy.stats as sp_stats

        col_embedder = self.col_embedder
        isab_blocks = col_embedder.tf_col.blocks

        n, p = X.shape if isinstance(X, np.ndarray) else X.shape
        if isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = X.cpu().numpy()
        X_np = X_np.astype(np.float32)


        gaussian_quantiles = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        X_processed = np.zeros((n, p), dtype=np.float32)
        for j in range(p):
            ranks = sp_stats.rankdata(X_np[:, j], method='average')
            X_processed[:, j] = gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]

        valid_layers = sorted(l for l in layers if l <= len(isab_blocks))
        results = {l: torch.zeros(n, p, self.embed_dim, device=self.device) for l in valid_layers}
        n_cls = self.n_cls_tokens
        fpg = self.feature_group_size

        col_embedder.eval()
        with torch.no_grad():
            for start in range(0, p, batch_size):
                end = min(start + batch_size, p)
                batch_p = end - start

                x_input = np.zeros((batch_p, 1 + n_cls, n, fpg), dtype=np.float32)
                if fpg == 1:
                    x_input[:, n_cls, :, 0] = X_processed[:, start:end].T
                else:
                    for i in range(fpg):
                        x_input[:, n_cls, :, i] = X_processed[:, start:end].T
                x_tensor = torch.as_tensor(x_input, device=self.device)


                h0 = col_embedder.in_linear(x_tensor)


                for layer in valid_layers:
                    block = isab_blocks[layer - 1]
                    h = block(h0)
                    emb = h[:, n_cls:, :, :].squeeze(1).permute(1, 0, 2)
                    results[layer][:, start:end, :] = emb.detach()

        return results

    def get_random_transformer_embeddings(self, X, layers, seed=42):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")

        col_embedder = self.col_embedder
        isab_blocks = col_embedder.tf_col.blocks


        saved_states = []
        for block in isab_blocks:
            saved_states.append({k: v.clone() for k, v in block.state_dict().items()})


        for block in isab_blocks:
            _randomize_module_params(block, seed=seed)

        col_embedder.eval()
        n, p = X.shape if isinstance(X, np.ndarray) else X.shape
        if isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = X.cpu().numpy()
        X_np = X_np.astype(np.float32)

        valid_layers = sorted(l for l in layers if l <= len(isab_blocks))
        results = {l: torch.zeros(n, p, self.embed_dim, device=self.device) for l in valid_layers}
        batch_size_cols = 32


        import scipy.stats as sp_stats
        gaussian_quantiles = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        X_processed = np.zeros((n, p), dtype=np.float32)
        for j in range(p):
            ranks = sp_stats.rankdata(X_np[:, j], method='average')
            X_processed[:, j] = gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]

        n_cls = self.n_cls_tokens
        fpg = self.feature_group_size

        with torch.no_grad():
            for start in range(0, p, batch_size_cols):
                end = min(start + batch_size_cols, p)
                batch_p = end - start

                x_input = np.zeros((batch_p, 1 + n_cls, n, fpg), dtype=np.float32)
                if fpg == 1:
                    x_input[:, n_cls, :, 0] = X_processed[:, start:end].T
                else:
                    for i in range(fpg):
                        x_input[:, n_cls, :, i] = X_processed[:, start:end].T
                x_tensor = torch.as_tensor(x_input, device=self.device)


                h0 = col_embedder.in_linear(x_tensor)


                for layer in valid_layers:
                    block = isab_blocks[layer - 1]
                    h = block(h0)
                    emb = h[:, n_cls:, :, :].squeeze(1).permute(1, 0, 2)
                    results[layer][:, start:end, :] = emb.detach()


        for block, state in zip(isab_blocks, saved_states):
            block.load_state_dict(state)

        return results

    def get_cumulative_layer_embeddings(self, X, layers, batch_size=32):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")
        import scipy.stats as sp_stats
        col_embedder = self.col_embedder
        isab_blocks = col_embedder.tf_col.blocks
        n, p = X.shape if isinstance(X, np.ndarray) else X.shape
        X_np = (X if isinstance(X, np.ndarray) else X.cpu().numpy()).astype(np.float32)

        gaussian_quantiles = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        X_processed = np.zeros((n, p), dtype=np.float32)
        for j in range(p):
            ranks = sp_stats.rankdata(X_np[:, j], method='average')
            X_processed[:, j] = gaussian_quantiles[np.clip(ranks.astype(int) - 1, 0, n - 1)]
        valid_layers = sorted(l for l in layers if l <= len(isab_blocks))
        results = {l: torch.zeros(n, p, self.embed_dim, device=self.device) for l in valid_layers}
        n_cls = self.n_cls_tokens
        fpg = self.feature_group_size
        max_layer = max(valid_layers)
        layer_set = set(valid_layers)
        col_embedder.eval()
        with torch.no_grad():
            for start in range(0, p, batch_size):
                end = min(start + batch_size, p)
                x_input = np.zeros((end - start, 1 + n_cls, n, fpg), dtype=np.float32)
                if fpg == 1:
                    x_input[:, n_cls, :, 0] = X_processed[:, start:end].T
                else:
                    for i in range(fpg):
                        x_input[:, n_cls, :, i] = X_processed[:, start:end].T
                x_tensor = torch.as_tensor(x_input, device=self.device)
                h = col_embedder.in_linear(x_tensor)
                for block_idx in range(max_layer):
                    h = isab_blocks[block_idx](h)
                    layer_num = block_idx + 1
                    if layer_num in layer_set:
                        emb = h[:, n_cls:, :, :].squeeze(1).permute(1, 0, 2)
                        results[layer_num][:, start:end, :] = emb.detach()
        return results

    def get_random_cumulative_embeddings(self, X, layers, batch_size=32, seed=42):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")
        col_embedder = self.col_embedder
        isab_blocks = col_embedder.tf_col.blocks
        saved_states = [{k: v.clone() for k, v in b.state_dict().items()} for b in isab_blocks]
        for b in isab_blocks:
            _randomize_module_params(b, seed=seed)
        results = self.get_cumulative_layer_embeddings(X, layers, batch_size=batch_size)
        for b, state in zip(isab_blocks, saved_states):
            b.load_state_dict(state)
        return results


class TabPFNColumnEmbedder:

    def __init__(self, model_path: str = None, device: str = 'cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.emsize = None
        self.features_per_group = None
        self.num_layers = None
        self.loaded = False
        self._model_path = model_path

    def load_model(self, model_path: str = None, seed: int = 42, device: str = None) -> bool:
        if self.loaded:
            return True
        if model_path is None:
            model_path = self._model_path
        if device is not None:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})
            architecture_name = checkpoint.get('architecture_name', 'base')

            from tabpfn.architectures import ARCHITECTURES
            arch_module = ARCHITECTURES.get(architecture_name)
            if arch_module is None:
                raise ValueError(f"Unknown architecture: {architecture_name}")

            parsed_config, _ = arch_module.parse_config(config)

            self.model = arch_module.get_architecture(
                parsed_config,
                cache_trainset_representation=False,
            )
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.emsize = parsed_config.emsize
            self.features_per_group = parsed_config.features_per_group
            self.num_layers = parsed_config.nlayers

            self.loaded = True
            print(f"TabPFNColumnEmbedder loaded: arch={architecture_name}, "
                  f"emsize={self.emsize}, fpg={self.features_per_group}, "
                  f"nlayers={self.num_layers}", flush=True)
            return True
        except Exception as e:
            print(f"TabPFNColumnEmbedder failed to load: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _preprocess_features(self, X: np.ndarray, y: np.ndarray):
        device = self.device
        n, p = X.shape


        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(1)  # (n, B=1, p)

        y_tensor = torch.zeros(n, device=device, dtype=torch.float32)


        model = self.model


        is_v2_5 = hasattr(model, 'blocks')  # TabPFNV2p5
        is_base = hasattr(model, 'transformer_encoder')  # PerFeatureTransformer

        with torch.no_grad():
            if is_v2_5:
                return self._preprocess_v2_5(x_tensor, y_tensor, n, p)
            elif is_base:
                return self._preprocess_base(x_tensor, y_tensor, n, p)
            else:
                raise ValueError(f"Unknown model architecture: {type(model)}")

    def _preprocess_base(self, x_tensor, y_tensor, n, p):
        import einops
        model = self.model
        device = self.device
        batch_size = 1

        x = {"main": x_tensor}
        y = {"main": y_tensor}

        seq_len, batch_size_check, num_features = x["main"].shape
        single_eval_pos = n

        # Pad features to multiple of features_per_group
        fpg = self.features_per_group
        missing = (fpg - (num_features % fpg)) % fpg
        if missing > 0:
            x["main"] = torch.cat((
                x["main"],
                torch.zeros(seq_len, batch_size, missing, device=device, dtype=x["main"].dtype),
            ), dim=-1)

        # Rearrange into feature groups: (s b f*fpg) -> (b s f fpg)
        x["main"] = einops.rearrange(x["main"], 's b (f n) -> b s f n', n=fpg)

        # Process y
        if y["main"].ndim == 1:
            y["main"] = y["main"].unsqueeze(-1)
        if y["main"].ndim == 2:
            y["main"] = y["main"].unsqueeze(-1)
        y["main"] = y["main"].transpose(0, 1)  # s b 1 -> b s 1

        # Pad y if needed
        if y["main"].shape[1] < x["main"].shape[1]:
            y["main"] = torch.cat((
                y["main"],
                torch.nan * torch.zeros(
                    y["main"].shape[0], x["main"].shape[1] - y["main"].shape[1],
                    y["main"].shape[2], device=device, dtype=y["main"].dtype,
                ),
            ), dim=1)
        y["main"] = y["main"].transpose(0, 1)  # b s 1 -> s b 1
        y["main"][single_eval_pos:] = torch.nan

        # Encode y
        embedded_y = model.y_encoder(
            y, single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
        ).transpose(0, 1)

        # Encode x
        for k in x:
            x[k] = einops.rearrange(x[k], 'b s f n -> s (b f) n')

        embedded_x = einops.rearrange(
            model.encoder(
                x, single_eval_pos=single_eval_pos,
                cache_trainset_representation=False,
            ),
            's (b f) e -> b s f e',
            b=embedded_y.shape[0],
        )

        # Add positional embeddings
        embedded_x, embedded_y = model.add_embeddings(
            embedded_x, embedded_y,
            data_dags=None, num_features=num_features, seq_len=seq_len,
            cache_embeddings=False, use_cached_embeddings=False,
        )

        # Concatenate: (b, s, f, e) + (b, s, 1, e) -> (b, s, f+1, e)
        embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        # Add thinking tokens
        if model.add_thinking_tokens is not None:
            embedded_input, single_eval_pos_updated = model.add_thinking_tokens(
                embedded_input, single_eval_pos=single_eval_pos,
            )
        else:
            single_eval_pos_updated = single_eval_pos

        return embedded_input, single_eval_pos_updated, num_features

    def _preprocess_v2_5(self, x_tensor, y_tensor, n, p):
        model = self.model
        device = self.device

        num_train_rows, batch_size, num_features = x_tensor.shape
        single_eval_pos = n


        embedded_x, _, _ = model._preprocess_and_embed_features(
            x_RiBC=x_tensor,
            num_train_labels=single_eval_pos,
            batch_size=batch_size,
        )

        embedded_y = model._preprocess_and_embed_targets(
            y=y_tensor,
            num_train_rows=num_train_rows,
            num_train_labels=single_eval_pos,
            batch_size=batch_size,
        )

        # Concatenate
        embedded_input = torch.cat((embedded_x, embedded_y[:, :, None]), dim=2)

        # Add thinking rows
        embedded_input, single_eval_pos_updated = model.add_thinking_rows(
            embedded_input, single_eval_pos=single_eval_pos,
        )

        return embedded_input, single_eval_pos_updated, num_features

    def get_embeddings(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        result = self.get_embeddings_multi_layer(X, y, layers=[self.num_layers])
        return result[self.num_layers]

    def get_embeddings_multi_layer(self, X: np.ndarray, y: np.ndarray,
                                    layers: List[int] = None) -> Dict[int, torch.Tensor]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        device = self.device
        n, p = X.shape

        embedded_input, single_eval_pos, num_features = self._preprocess_features(X, y)

        if layers is None:
            layers = [self.num_layers]

        layer_set = set(layers)
        max_layer = max(layers) if layers else self.num_layers


        results = {}
        save_peak_mem_factor = None
        is_v2_5 = hasattr(self.model, 'blocks')

        with torch.no_grad():
            if is_v2_5:
                is_v2_6 = isinstance(self.model, type(self.model)) and type(self.model).__name__ == 'TabPFNV2p6'
                for layer_num in sorted(layers):
                    if layer_num > len(self.model.blocks):
                        continue
                    block = self.model.blocks[layer_num - 1]
                    x = embedded_input.clone()
                    x_bcre = x.transpose(1, 2).contiguous()
                    if is_v2_6:
                        # v2.6 expects (Bc, R, E) = (B*C, R, E), 3D

                        B, C, R, E = x_bcre.shape
                        x_bcre_flat = x_bcre.reshape(B * C, R, E)
                        attn_out, _ = block.per_column_attention_between_cells(
                            x_bcre_flat, single_eval_pos=single_eval_pos
                        )
                        x_bcre_flat = attn_out + x_bcre_flat
                        x_bcre_flat = block.layernorm_mha2(x_bcre_flat)
                        x_bcre = x_bcre_flat.reshape(B, C, R, E)
                    else:
                        # v2.5 expects (B, C, R, E), 4D
                        attn_out, _ = block.per_column_attention_between_cells(
                            x_bcre, single_eval_pos=single_eval_pos
                        )
                        x_bcre = attn_out + x_bcre
                        x_bcre = block.layernorm_mha2(x_bcre)
                    x = x_bcre.transpose(1, 2).contiguous()
                    x = block.mlp(x) + x
                    x = block.layernorm_mlp(x)
                    results[layer_num] = self._extract_feature_embeddings(
                        x, single_eval_pos, n, p)
            else:

                all_layers = list(self.model.transformer_encoder.layers)
                for layer_num in sorted(layers):
                    if layer_num > len(all_layers):
                        continue
                    layer = all_layers[layer_num - 1]
                    x = embedded_input.clone()

                    sublayer_list = []
                    if layer.self_attn_between_features is not None:
                        sublayer_list.append(('_skip_feat_attn', None))

                    def _make_item_attn(layer_ref, sep):
                        def _item_attn(state):
                            src = state[:, :sep].transpose(1, 2) if sep else None
                            out = layer_ref.self_attn_between_items(
                                state.transpose(1, 2),
                                src,
                                save_peak_mem_factor=None,
                                cache_kv=False,
                                add_input=True,
                                allow_inplace=True,
                                use_cached_kv=False,
                            )
                            return out.transpose(1, 2)
                        return _item_attn

                    sublayer_list.append(('item_attn', _make_item_attn(layer, single_eval_pos)))
                    sublayer_list.append(('mlp', lambda s, lr=layer: lr.mlp(
                        s, add_input=True, allow_inplace=True)))

                    if layer.second_mlp is not None:
                        sublayer_list.insert(1, ('second_mlp', lambda s, lr=layer: lr.second_mlp(
                            s, add_input=True, allow_inplace=True)))

                    for idx, (name, fn) in enumerate(sublayer_list):
                        if name == '_skip_feat_attn':
                            continue
                        x = fn(x)
                        x = layer.layer_norms[idx](
                            x, allow_inplace=True, save_peak_mem_factor=None)

                    results[layer_num] = self._extract_feature_embeddings(
                        x, single_eval_pos, n, p)

        return results

    def get_random_embeddings_multi_layer(self, X: np.ndarray, y: np.ndarray,
                                           layers: List[int] = None,
                                           seed: int = 42) -> Dict[int, torch.Tensor]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        model = self.model
        is_v2_5 = hasattr(model, 'blocks')


        if is_v2_5:
            blocks = model.blocks
            saved_states = [{k: v.clone() for k, v in b.state_dict().items()} for b in blocks]

            for b in blocks:
                _randomize_module_params(b, seed=seed)
        else:
            layers_list = model.transformer_encoder.layers
            saved_states = [{k: v.clone() for k, v in l.state_dict().items()} for l in layers_list]

            for l in layers_list:
                _randomize_module_params(l, seed=seed)


        results = self.get_embeddings_multi_layer(X, y, layers=layers)


        if is_v2_5:
            for b, state in zip(blocks, saved_states):
                b.load_state_dict(state)
        else:
            for l, state in zip(layers_list, saved_states):
                l.load_state_dict(state)

        return results

    def get_projection_embeddings(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        device = self.device
        model = self.model
        n, p = X.shape
        emsize = self.emsize
        fpg = self.features_per_group

        is_v2_5 = hasattr(model, 'blocks')

        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        results = torch.zeros(n, p, emsize, device=device)

        CHUNK_SIZE = 500

        with torch.no_grad():
            col_emb = self._compute_column_embeddings(p, device)

            for start in range(0, p, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, p)
                chunk_p = end - start
                x_chunk = x_tensor[:, start:end]

                x_repeated = x_chunk.unsqueeze(2).expand(-1, -1, fpg).reshape(n, chunk_p, fpg)

                if is_v2_5:
                    embedded_x = self._proj_v2_5(x_repeated, n, chunk_p)
                else:
                    embedded_x = self._proj_base(x_repeated, n, chunk_p)

                results[:, start:end, :] = embedded_x + col_emb[None, start:end, :]

        return results

    def _proj_v2_5(self, x_RiBgF, n, chunk_p):
        from tabpfn.architectures.tabpfn_v2_5 import (
            _generate_nan_and_inf_indicator,
            _impute_nan_and_inf_with_mean,
            _normalize_feature_groups,
        )
        model = self.model
        fpg = self.features_per_group

        nan_indicator = _generate_nan_and_inf_indicator(x=x_RiBgF)
        x_RiBgF, _ = _impute_nan_and_inf_with_mean(x=x_RiBgF, num_train_rows=n)
        x_RiBgF = model.standard_scaler(x=x_RiBgF, num_train_rows=n)

        x_RiBgF = _normalize_feature_groups(x_RiBgF, num_features_per_group=fpg)[0]

        x_concat = torch.cat([x_RiBgF, nan_indicator], dim=-1)
        return model.feature_group_embedder(x_concat)

    def _proj_base(self, x_RiGpF, n, chunk_p):
        model = self.model
        embedded_x = model.encoder(x_RiGpF, single_eval_pos=n)
        return embedded_x

    def _compute_column_embeddings(self, num_positions: int, device: torch.device) -> torch.Tensor:
        model = self.model
        emsize = self.emsize
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(42)
            embs = torch.randn(
                (num_positions, emsize // 4),
                device=device, dtype=torch.float32,
                generator=generator,
            )
            if embs.shape[1] == model.pre_generated_column_embeddings.shape[1]:
                n_pre = min(num_positions, model.pre_generated_column_embeddings.shape[0])
                embs[:n_pre] = model.pre_generated_column_embeddings[:n_pre].to(
                    device=embs.device, dtype=embs.dtype,
                )
            embs = model.feature_positional_embedding_embeddings(embs)
        return embs

    def _extract_feature_embeddings(self, tensor, single_eval_pos, n, p):
        device = self.device
        fpg = self.features_per_group
        num_groups = (p + fpg - 1) // fpg

        thinking_rows = 0
        if hasattr(self.model, 'add_thinking_tokens') and self.model.add_thinking_tokens is not None:
            thinking_rows = self.model.add_thinking_tokens.num_thinking_rows
        elif hasattr(self.model, 'add_thinking_rows') and self.model.add_thinking_rows is not None:
            thinking_rows = self.model.add_thinking_rows.num_thinking_rows

        train_start = thinking_rows
        train_end = single_eval_pos

        num_cols = tensor.shape[2]
        feature_cols = min(num_groups, num_cols - 1)

        group_emb = tensor[0, train_start:train_end, :feature_cols, :].detach()

        results = torch.zeros(n, p, self.emsize, device=device)
        for g in range(num_groups):
            feat_start = g * fpg
            feat_end = min(feat_start + fpg, p)
            for j in range(feat_start, feat_end):
                results[:, j, :] = group_emb[:, g, :]

        return results


class TabPFNV3ColumnEmbedder:

    def __init__(self, model_path: str = None, device: str = 'cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embed_dim = None
        self.feature_group_size = None
        self.use_nan_indicators = None
        self.num_layers = None
        self.loaded = False
        self._model_path = model_path

    def load_model(self, model_path: str = None, seed: int = 42, device: str = None) -> bool:
        if self.loaded:
            return True
        if model_path is None:
            model_path = self._model_path
        if device is not None:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})
            architecture_name = checkpoint.get('architecture_name', 'tabpfn_v3')
            from tabpfn.architectures import ARCHITECTURES
            arch_module = ARCHITECTURES.get(architecture_name)
            if arch_module is None:
                raise ValueError(f"Unknown architecture: {architecture_name}")
            parsed_config, _ = arch_module.parse_config(config)
            try:
                self.model = arch_module.get_architecture(parsed_config, cache_trainset_representation=False)
            except Exception:

                task_type = config.get('task_type') or checkpoint.get('task_type') or 'multiclass'
                self.model = arch_module.get_architecture(parsed_config, cache_trainset_representation=False,
                                                          task_type=task_type)
            state_dict = checkpoint['state_dict']
            model_state = {k: v for k, v in state_dict.items() if not k.startswith('criterion.')}
            self.model.load_state_dict(model_state, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embed_dim = self.model.x_embed.out_features
            self.feature_group_size = self.model.feature_group_size
            self.use_nan_indicators = self.model.use_nan_indicators
            self.num_layers = len(self.model.feature_distribution_embedder.layers)
            self.loaded = True
            print(f"TabPFNV3ColumnEmbedder loaded: embed_dim={self.embed_dim}, "
                  f"fpg={self.feature_group_size}, use_nan={self.use_nan_indicators}, "
                  f"dist_blocks={self.num_layers}", flush=True)
            return True
        except Exception as e:
            print(f"TabPFNV3ColumnEmbedder failed to load: {e}", flush=True)
            import traceback; traceback.print_exc()
            return False


    def _preprocess_single_feature(self, X):
        device = self.device
        n, p = X.shape
        x_RiBC = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device).unsqueeze(1)  # (n,1,p)
        num_train = n
        with torch.no_grad():
            x_BRiC, nan_ind_BRiC, _ = self.model._preprocess_raw(x_RiBC, num_train=num_train)  # (1,n,p),(1,n,p)|None
            fpg = self.feature_group_size

            x_grouped = torch.stack([x_BRiC] * fpg, dim=-1)  # (1,n,p,fpg)
            if self.use_nan_indicators and nan_ind_BRiC is not None:
                ind_grouped = torch.stack([nan_ind_BRiC] * fpg, dim=-1)  # (1,n,p,fpg)
                x_grouped = torch.cat([x_grouped, ind_grouped], dim=-1)  # (1,n,p,fpg*2)
        return x_grouped, n, p

    def _proj_from_grouped(self, x_grouped):
        with torch.no_grad():
            return self.model.x_embed(x_grouped)

    def _extract(self, X, layers, mode, col_chunk=4096):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")
        x_grouped, n, p = self._preprocess_single_feature(X)
        valid_layers = sorted(l for l in layers if l <= self.num_layers)
        results = {l: torch.zeros(n, p, self.embed_dim, device=self.device) for l in valid_layers}
        layers_mod = self.model.feature_distribution_embedder.layers
        with torch.no_grad():
            for start in range(0, p, col_chunk):
                end = min(start + col_chunk, p)
                xg_chunk = x_grouped[:, :, start:end, :]  # (1,n,pc,G)
                h0 = self.model.x_embed(xg_chunk)          # (1,n,pc,E)
                if mode == 'independent':
                    for layer in valid_layers:
                        out, _ = layers_mod[layer - 1](h0, single_eval_pos=n)
                        results[layer][:, start:end, :] = out.squeeze(0).detach()
                else:  # cumulative
                    h = h0
                    for block_idx in range(max(valid_layers)):
                        h, _ = layers_mod[block_idx](h, single_eval_pos=n)
                        layer_num = block_idx + 1
                        if layer_num in valid_layers:
                            results[layer_num][:, start:end, :] = h.squeeze(0).detach()
        return results

    def get_projection_embeddings(self, X, col_chunk=4096):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")
        x_grouped, n, p = self._preprocess_single_feature(X)
        results = torch.zeros(n, p, self.embed_dim, device=self.device)
        with torch.no_grad():
            for start in range(0, p, col_chunk):
                end = min(start + col_chunk, p)
                out = self.model.x_embed(x_grouped[:, :, start:end, :])
                results[:, start:end, :] = out.squeeze(0).detach()
        return results

    def get_single_layer_embeddings(self, X, layers):
        return self._extract(X, layers, mode='independent')

    def get_cumulative_layer_embeddings(self, X, layers):
        return self._extract(X, layers, mode='cumulative')

    def get_random_embeddings_multi_layer(self, X, layers, mode='independent', seed=42):
        if not self.loaded:
            raise RuntimeError("Model not loaded.")
        layers_mod = self.model.feature_distribution_embedder.layers
        saved = [{k: v.clone() for k, v in b.state_dict().items()} for b in layers_mod]
        for b in layers_mod:
            _randomize_module_params(b, seed=seed)
        results = self._extract(X, layers, mode=mode)
        for b, st in zip(layers_mod, saved):
            b.load_state_dict(st)
        return results

