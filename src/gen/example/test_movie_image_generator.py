import unittest
import torch
from movie_image_generator import Generator, Discriminator, LATENT_DIM, IMAGE_SIZE

class TestGANModels(unittest.TestCase):
    
    def setUp(self):
        self.latent_vector = torch.randn(1, LATENT_DIM, 1, 1)
        self.fake_image_shape = (1, 1, IMAGE_SIZE, IMAGE_SIZE)
        self.real_image = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    
    def test_generator_output_shape(self):
        netG = Generator()
        output = netG(self.latent_vector)
        self.assertEqual(output.shape, self.fake_image_shape,
                         f"Generator output shape should be {self.fake_image_shape} but got {output.shape}")
    
    def test_generator_output_range(self):
        netG = Generator()
        output = netG(self.latent_vector)
        self.assertTrue(torch.all(output <= 1.0) and torch.all(output >= -1.0),
                        "Generator output should be in range [-1, 1] due to Tanh activation")
    
    def test_discriminator_output_shape(self):
        netD = Discriminator()
        output = netD(self.real_image)
        self.assertEqual(output.shape, torch.Size([1]),
                         f"Discriminator output should be a scalar value per image but got {output.shape}")
    
    def test_discriminator_output_range(self):
        netD = Discriminator()
        output = netD(self.real_image)
        self.assertTrue((0 <= output.item() <= 1), 
                        "Discriminator output should be a probability between 0 and 1")

    def test_training_step_shapes(self):
        netG = Generator()
        netD = Discriminator()
        criterion = torch.nn.BCELoss()
        optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        noise = torch.randn(1, LATENT_DIM, 1, 1)
        fake = netG(noise)
        label_real = torch.ones(1)
        label_fake = torch.zeros(1)

        # Train Discriminator
        optimizerD.zero_grad()
        output_real = netD(self.real_image)
        loss_real = criterion(output_real, label_real)

        output_fake = netD(fake.detach())
        loss_fake = criterion(output_fake, label_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # Train Generator
        optimizerG.zero_grad()
        output = netD(fake)
        loss_G = criterion(output, label_real)
        loss_G.backward()
        optimizerG.step()

        self.assertGreater(loss_D.item(), 0, "Discriminator loss should be greater than 0")
        self.assertGreater(loss_G.item(), 0, "Generator loss should be greater than 0")

if __name__ == '__main__':
    unittest.main()
