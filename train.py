import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Discriminator, Generator
from dataset import WavDataset, MP3Dataset


def train(device,
          wav_loader,
          mp3_loader,
          generator,
          discriminator,
          optimizer_G,
          optimizer_D,
          gradient_penalty_constant,
          discriminator_updates_per_generator_update,
          num_epochs,
          save_path):
    for epoch in range(num_epochs):
        for i, (mp3, wav) in enumerate(zip(mp3_loader, wav_loader)):
            mp3, wav = mp3.to(device), wav.to(device)
            for _ in range(discriminator_updates_per_generator_update):
                real_labels = torch.ones((mp3.size(0), 1)).to(device)
                fake_labels = torch.zeros((mp3.size(0), 1)).to(device)

                # Train the discriminator with real data
                outputs_real = discriminator(wav)
                d_loss_real = -torch.mean(outputs_real)

                # Train the discriminator with fake data
                fake_images = generator(mp3)
                outputs_fake = discriminator(fake_images)
                d_loss_fake = torch.mean(outputs_fake)

                # Compute the gradient penalty
                alpha = torch.rand(mp3.size(0), 1, 1, 1).to(device)
                interpolates = alpha * wav + (1 - alpha) * fake_images
                interpolates.requires_grad_(True)
                disc_interpolates = discriminator(interpolates)
                gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                d_loss = d_loss_real + d_loss_fake + gradient_penalty_constant * gradient_penalty

                discriminator.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            # Generator updates
            fake_images = generator(mp3)
            outputs = discriminator(fake_images)
            g_loss = -torch.mean(outputs)

            generator.zero_grad()
            g_loss.backward()
            optimizer_G.step()
        torch.save(generator.state_dict(), save_path + 'bestDQTGAN_{}.pth'.format(epoch))
        print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.6f}, g_loss: {g_loss.item():.6f}')


def main():
    device = "cuda"
    batch_size = 4
    discriminator_lr = 1e-5
    generator_lr = 5e-6
    num_epochs = 100
    gradient_penalty_constant = 1
    discriminator_updates_per_generator_update = 1

    wav_dataset = WavDataset(wav_path='data_folder/wav_data')
    mp3_dataset = MP3Dataset(mp3_path='data_folder/mp3_data')

    wav_loader = DataLoader(wav_dataset, batch_size=batch_size, shuffle=False)
    mp3_loader = DataLoader(mp3_dataset, batch_size=batch_size, shuffle=False)

    generator = Generator().to(device)
    generator.load_state_dict(torch.load('ckpt/bestDQTGAN_100.pth'))
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=discriminator_lr)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=generator_lr)

    save_path = 'ckpt/'

    print("Start Training...")
    train(device,
          wav_loader,
          mp3_loader,
          generator,
          discriminator,
          optimizer_G,
          optimizer_D,
          gradient_penalty_constant,
          discriminator_updates_per_generator_update,
          num_epochs,
          save_path)


if __name__ == "__main__":
    main()
