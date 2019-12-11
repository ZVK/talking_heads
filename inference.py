import torch
import cv2
from matplotlib import pyplot as plt
from network.model import Generator
from video_extraction import generate_landmarks
import os.path


def main(model_weights_path: str = 'model_weights.tar',
         embedding_path: str = 'e_hat_video.tar',
         video_path: str = 'examples/fine_tuning/test_video.mp4',
         output_dir: str = './'):
    """Init"""
    device = torch.device("cuda:0")
    cpu = torch.device("cpu")

    checkpoint = torch.load(model_weights_path, map_location=cpu)
    e_hat = torch.load(embedding_path, map_location=cpu)
    e_hat = e_hat['e_hat'].to(device)

    generator = Generator(256, finetuning=True, e_finetuning=e_hat)
    generator.eval()

    """Training Init"""
    generator.load_state_dict(checkpoint['G_state_dict'])
    generator.to(device)
    generator.finetuning_init()

    """Main"""
    print('PRESS Q TO EXIT')
    cap = cv2.VideoCapture(video_path)

    with torch.no_grad():
        enum = 0
        while True:
            print("doing enum", enum)
            x, g_y = generate_landmarks(cap=cap, device=device, pad=50)
            if x is None and g_y is None:
                print("broke at enum ", enum)
                break
            g_y = g_y.unsqueeze(0)
            x = x.unsqueeze(0)

            # forward
            # Calculate average encoding vector for video
            # f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224
            # train generator

            x_hat = generator(g_y, e_hat)

            plt.clf()
            out1 = x_hat.transpose(1, 3)[0] / 255
            for img_no in range(1,x_hat.shape[0]):
               out1 = torch.cat((out1, x_hat.transpose(1,3)[img_no]), dim = 1)
            out1 = out1.to(cpu).numpy()
            plt.imshow(out1)
            plt.show()
            plt.imsave(os.path.join(output_dir, 'fake-{}.png'.format(enum)), out1)
            plt.clf()
            out2 = x.transpose(1, 3)[0] / 255
            for img_no in range(1,x.shape[0]):
               out2 = torch.cat((out2, x.transpose(1,3)[img_no]), dim = 1)
            out2 = out2.to(cpu).numpy()
            plt.imshow(out2)
            plt.show()
            plt.imsave(os.path.join(output_dir, 'head_track-{}.png'.format(enum)), out2)
            plt.clf()
            out3 = g_y.transpose(1, 3)[0] / 255
            for img_no in range(1,g_y.shape[0]):
               out3 = torch.cat((out3, g_y.transpose(1,3)[img_no]), dim = 1)
            out3 = out3.to(cpu).numpy()
            plt.imshow(out3)
            plt.show()
            plt.imsave(os.path.join(output_dir, 'landmark-{}.png'.format(enum)), out3)
            plt.clf()
            if cv2.waitKey(1) == ord('q'):
                break
            enum += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
