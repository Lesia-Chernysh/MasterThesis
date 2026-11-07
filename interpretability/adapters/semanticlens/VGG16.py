from template import ActivationComponentVisualizerAdapter, ActivationCachingAdapter

class ActCachingAdapterVGG16(ActivationCachingAdapter):
    def create_acts_not_aggr_shape(self, n_collect, n_all_latents):
        acts_not_aggr_shape = [n_collect] + list(n_all_latents)  # --> [n_images, n_neurons, *h, *w]
        acts_not_aggr_shape = [acts_not_aggr_shape[1], acts_not_aggr_shape[0]] + acts_not_aggr_shape[2:]
        return acts_not_aggr_shape

    def create_n_all_latents(self, acts_not_aggr):
        n_all_latents = list(acts_not_aggr.shape[1:])  # the first dim is n of images
        return n_all_latents

    def create_batch_acts_not_aggr(self, acts_not_aggr, activations_not_aggr, device):
        # [imgs, n_neurons, h, w] --> [n_neurons, imgs h, w]
        batch_acts_not_aggr = acts_not_aggr.transpose(1, 0).to(activations_not_aggr.dtype).to(device)
        return batch_acts_not_aggr

    def create_indices_reshaped(self, n_all_latents, indices, device):
        h, w = n_all_latents[-2], n_all_latents[-1]
        indices_reshaped = indices.unsqueeze(-1).unsqueeze(-1).to(device)  # resnet: [1024, 10, 1, 1]
        indices_reshaped = indices_reshaped.expand(-1, -1, h, w)  # resnet [1024, 10, 14, 14]
        return indices_reshaped

class CVAdapterVGG16(ActivationComponentVisualizerAdapter):
    def construct_input(self, images, device, *args):
        return images.to(device)

    def create_heatmap2d(self, actmax_cache, layer, neuron_id, img_pos):
        # heatmap_2d
        #   vgg16:  [imgs, hid=512,  channl_h, channl_w]
        heatmap_2d = actmax_cache.cache[layer].activations_not_aggr[neuron_id, img_pos].cpu().float().numpy()
        grid_h, grid_w = heatmap_2d.shape[-2], heatmap_2d.shape[-1]

        return heatmap_2d, grid_h, grid_w