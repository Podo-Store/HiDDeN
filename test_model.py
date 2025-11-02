import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration
import hashlib

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')
    parser.add_argument('--message-int', type=str, required=False,
                        help='Integer value to encode as message bits (auto-detects base like 0x..., 0b...)')
    parser.add_argument('--use-hash', action='store_true',
                        help='Hash the provided --message-int string with SHA-256 and map to message_length bits')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    noiser = Noiser(noise_config, device)

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)


    image_pil = Image.open(args.source_image)
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    image_tensor.unsqueeze_(0)

    # for t in range(args.times):
    L = hidden_config.message_length
    B = image_tensor.shape[0]
    if args.message_int is not None:
        if args.use_hash:
            h = hashlib.sha256(args.message_int.encode('utf-8')).digest()
            bits = []
            for b in h:
                for i in range(7, -1, -1):
                    bits.append((b >> i) & 1)
            if len(bits) < L:
                bits = bits + [0] * (L - len(bits))
            else:
                bits = bits[:L]
            message = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
        else:
            try:
                val = int(args.message_int, 0)
            except Exception:
                val = int(args.message_int)
            mask = (1 << L) - 1
            val = val & mask
            bits = [(val >> i) & 1 for i in range(L - 1, -1, -1)]
            message = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
    else:
        message = torch.Tensor(np.random.choice([0, 1], (B, L))).to(device)
    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    message_detached = message.detach().cpu().numpy()
    print('original: {}'.format(message_detached))
    print('decoded : {}'.format(decoded_rounded))
    print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))
    decoded_bits = decoded_rounded.astype(np.int32)
    orig_bits = message_detached.astype(np.int32)
    mismatch = (decoded_bits != orig_bits).astype(np.int32)
    mismatch_idx = np.where(mismatch[0] == 1)[0]
    print('mismatch_count: {} / {}'.format(int(mismatch.sum()), orig_bits.shape[1]))
    print('mismatch_idx  : {}'.format(mismatch_idx.tolist()))
    s_orig = ''.join(str(x) for x in orig_bits[0].tolist())
    s_dec = ''.join(str(x) for x in decoded_bits[0].tolist())
    s_mark = ''.join('^' if m == 1 else ' ' for m in mismatch[0].tolist())
    print('orig_bits:', s_orig)
    print('dec_bits :', s_dec)
    print('marker   :', s_mark)
    if args.message_int is not None and not args.use_hash:
        orig_val = int(s_orig, 2)
        dec_val = int(s_dec, 2)
        print('orig_int(masked {}b): {}'.format(L, orig_val))
        print('dec_int(masked {}b) : {}'.format(L, dec_val))
    utils.save_images(image_tensor.cpu(), encoded_images.cpu(), 'test', '.', resize_to=(256, 256))

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])



if __name__ == '__main__':
    main()
