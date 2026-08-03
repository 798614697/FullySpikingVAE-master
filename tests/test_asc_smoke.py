import torch

import global_v as glv
from attribute_classifier import AttributeClassifier
from fsvae_models.asc_fsvae import ASCFSVAELarge, normalize_attributes, spike_attribute_bce
from fsvae_models.fsvae import FSVAELarge


def tiny_config():
    return {'batch_size': 2, 'n_steps': 2, 'latent_dim': 4, 'k': 2,
            'in_channels': 3, 'condition_dim': 40, 'condition_embedding_dim': 3,
            'attribute_hidden_dim': 5, 'condition_population': 2,
            'scale_lr_by_batch': False, 'lr': 0.001}


def test_conditional_forward_sample_and_gradient():
    glv.init(tiny_config(), [0])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ASCFSVAELarge(use_encoder_attribute_head=True).to(device)
    # FSVAELarge is fixed at five downsampling stages, hence 64x64 input.
    images = torch.randn(2, 3, 64, 64, device=device)
    spikes = images.unsqueeze(-1).repeat(1, 1, 1, 1, 2)
    condition = torch.randint(0, 2, (2, 40), device=device).float()
    recon, q_z, p_z, sampled_z, attr_spikes = model(spikes, condition)
    assert recon.shape == images.shape
    assert q_z.shape == p_z.shape == (2, 4, 2, 2)
    assert sampled_z.shape == (2, 4, 2)
    assert attr_spikes.shape == (2, 40, 2, 2)
    attr_loss, probability = spike_attribute_bce(attr_spikes, condition,
                                                 torch.ones(40, device=device))
    (model.loss_function_mmd(images, recon, q_z, p_z)['loss'] + attr_loss).backward()
    assert model.condition_encoder.layer.weight.grad is not None
    assert model.attribute_head.layers[0].weight.grad is not None
    generated, generated_z = model.sample(condition)
    assert generated.shape == images.shape
    assert generated_z.shape == (2, 4, 2)
    try:
        model.sample(None)
    except ValueError:
        pass
    else:
        raise AssertionError('sample(None) must reject a missing condition')


def test_attribute_normalization_and_classifier_input_gradient():
    values = torch.tensor([[-1, 1], [1, -1]])
    assert torch.equal(normalize_attributes(values), torch.tensor([[0., 1.], [1., 0.]]))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = AttributeClassifier().to(device)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    image = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    classifier(image).sum().backward()
    assert image.grad is not None


def test_common_layers_have_identical_initialization():
    glv.init(tiny_config(), [0])
    torch.manual_seed(2024)
    baseline = FSVAELarge()
    torch.manual_seed(2024)
    conditional = ASCFSVAELarge()
    for prefix in ('encoder.', 'before_latent_layer.', 'decoder_input.',
                   'decoder.', 'final_layer.'):
        baseline_values = {name: value for name, value in baseline.state_dict().items()
                           if name.startswith(prefix)}
        conditional_values = {name: value for name, value in conditional.state_dict().items()
                              if name.startswith(prefix)}
        assert baseline_values.keys() == conditional_values.keys()
        assert all(torch.equal(value, conditional_values[name])
                   for name, value in baseline_values.items())
