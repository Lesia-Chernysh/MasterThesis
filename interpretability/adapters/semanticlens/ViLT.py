from transformers import ViltProcessor
from template import ActivationComponentVisualizerAdapter, ActivationCachingAdapter
from models.create_model import create_vilt

class ActCachingAdapterViLT(ActivationCachingAdapter):
    def create_acts_not_aggr_shape(self, n_collect, n_all_latents):
        acts_not_aggr_shape = [n_collect] + list(n_all_latents)  # --> [n_images, hid_d, n_neurons]
        acts_not_aggr_shape = [acts_not_aggr_shape[2], acts_not_aggr_shape[0], acts_not_aggr_shape[1]]
        return acts_not_aggr_shape

    def create_n_all_latents(self, acts_not_aggr):
        n_all_latents = list(acts_not_aggr.shape[1:])  # the first dim is n of images
        return n_all_latents

    def create_batch_acts_not_aggr(self, acts_not_aggr, activations_not_aggr, device):
        # [imgs, hidden, n_neurons] --> [n_neurons, imgs, hidden]
        batch_acts_not_aggr = acts_not_aggr.permute(2, 0, 1).to(activations_not_aggr.dtype).to(device)
        return batch_acts_not_aggr

    def create_indices_reshaped(self, n_all_latents, indices, device):
        hid = n_all_latents[0]  # n_all_latents = [hid, n_neurons]
        indices_reshaped = indices.unsqueeze(-1).to(device)  # [n_nueorns, 10, 1]
        indices_reshaped = indices_reshaped.expand(-1, -1, hid)
        return indices_reshaped

class CVAdapterViLT(ActivationComponentVisualizerAdapter):

    def construct_input(self, images, device,  *args):
        """
        :param images: a batch of images
        :param device: device of the CV class
        :param args:
        :return: encoding of images and question, as required as input to ViLT
        """
        # TODO: now, processor is initialized for each batch, which is dull. Where to initalize it instead?
        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        text = "What object is depicted on the image?"
        text_batch = [text] * len(images)
        encoding = processor(images=images.to(device), text=text_batch, return_tensors="pt", padding=True)

        return encoding

    def create_heatmap2d(self, actmax_cache, layer, neuron_id, img_pos):
        model = create_vilt()
        all_token_activations = (
            actmax_cache.cache[layer]
            .activations_not_aggr[neuron_id, img_pos, :]
            .cpu()
            .float()
            .numpy()
        )

        patch_h, patch_w = (
            model.vilt.embeddings
            .patch_embeddings.projection.kernel_size
        )

        grid_h, grid_w = int(224 / patch_h), int(224 / patch_w)  # --> 14, 14
        num_patches = grid_h * grid_w

        # Everything before the final image patches consists of text/special tokens.
        visual_offset = len(all_token_activations) - num_patches
        visual_patch_activations = all_token_activations[visual_offset:]

        heatmap_2d = visual_patch_activations.reshape(grid_h, grid_w)

        return heatmap_2d, grid_h, grid_w

