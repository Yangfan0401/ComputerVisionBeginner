
def rnn_test():
    # DEVICE = "cpu"
    # to_float = {"dtype": torch.float32, "device": DEVICE}
    # to_double = {"dtype": torch.float64, "device": DEVICE}

    # # Set a few constants related to data loading.
    # IMAGE_SHAPE = (112, 112)
    # NUM_WORKERS = 1

    # # Batch size used for full training runs:
    # BATCH_SIZE = 256

    # # Batch size used for overfitting sanity checks:
    # OVR_BATCH_SIZE = BATCH_SIZE // 8

    # # Batch size used for visualization:
    # VIS_BATCH_SIZE = 4

    # # Download and load serialized COCO data from coco.pt
    # # It contains a dictionary of
    # # "train_images" - resized training images (IMAGE_SHAPE)
    # # "val_images" - resized validation images (IMAGE_SHAPE)
    # # "train_captions" - tokenized and numericalized training captions
    # # "val_captions" - tokenized and numericalized validation captions
    # # "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

    # if os.path.isfile("./datasets/coco.pt"):
    #     print("COCO data exists!")
    # else:
    #     print("downloading COCO dataset")

    # # load COCO data from coco.pt, loaf_COCO is implemented in a5_helper.py
    # data_dict = load_coco_captions(path="./datasets/coco.pt")

    # num_train = data_dict["train_images"].size(0)
    # num_val = data_dict["val_images"].size(0)

    # # declare variables for special tokens
    # NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]
    # START_index = data_dict["vocab"]["token_to_idx"]["<START>"]
    # END_index = data_dict["vocab"]["token_to_idx"]["<END>"]
    # UNK_index = data_dict["vocab"]["token_to_idx"]["<UNK>"]

    # # # # Sample a minibatch and show the reshaped 112x112 images and captions
    # # # sample_idx = torch.randint(0, num_train, (VIS_BATCH_SIZE, ))
    # # # sample_images = data_dict["train_images"][sample_idx]
    # # # sample_captions = data_dict["train_captions"][sample_idx]
    # # # for i in range(VIS_BATCH_SIZE):
    # # #     plt.imshow(sample_images[i].permute(1, 2, 0))
    # # #     plt.axis("off")
    # # #     caption_str = decode_captions(
    # # #         sample_captions[i], data_dict["vocab"]["idx_to_token"]
    # # #     )
    # # #     plt.title(caption_str)
    # # #     plt.tight_layout()
    # # #     plt.show()
    # #
    # # # NOTE: 搭建RNN前向传播函数
    # # N, D, H = 3, 10, 4
    # #
    # # x = torch.linspace(-0.4, 0.7, steps=N * D, **to_double).view(N, D)
    # # prev_h = torch.linspace(-0.2, 0.5, steps=N * H, **to_double).view(N, H)
    # # Wx = torch.linspace(-0.1, 0.9, steps=D * H, **to_double).view(D, H)
    # # Wh = torch.linspace(-0.3, 0.7, steps=H * H, **to_double).view(H, H)
    # # b = torch.linspace(-0.2, 0.4, steps=H, **to_double)
    # #
    # #
    # # next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
    # # expected_next_h = torch.tensor(
    # #     [
    # #         [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
    # #         [0.66854692, 0.79562378, 0.87755553, 0.92795967],
    # #         [0.97934501, 0.99144213, 0.99646691, 0.99854353],
    # #     ],
    # #     **to_double,
    # # )
    # #
    # # print("next_h error: ", rel_error(expected_next_h, next_h))
    # #
    # #
    # # # NOTE: 搭建RNN反向传播函数
    # # reset_seed(0)
    # #
    # # N, D, H = 4, 5, 6
    # # x = torch.randn(N, D, **to_double)
    # # h = torch.randn(N, H, **to_double)
    # # Wx = torch.randn(D, H, **to_double)
    # # Wh = torch.randn(H, H, **to_double)
    # # b = torch.randn(H, **to_double)
    # #
    # # out, cache = rnn_step_forward(x, h, Wx, Wh, b)
    # #
    # # dnext_h = torch.randn(*out.shape, **to_double)
    # #
    # # fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
    # # fh = lambda h: rnn_step_forward(x, h, Wx, Wh, b)[0]
    # # fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
    # # fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
    # # fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]
    # #
    # # dx_num = compute_numeric_gradient(fx, x, dnext_h)
    # # dprev_h_num = compute_numeric_gradient(fh, h, dnext_h)
    # # dWx_num = compute_numeric_gradient(fWx, Wx, dnext_h)
    # # dWh_num = compute_numeric_gradient(fWh, Wh, dnext_h)
    # # db_num = compute_numeric_gradient(fb, b, dnext_h)
    # #
    # # # YOUR_TURN: Implement rnn_step_backward
    # # dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
    # #
    # # print("dx error: ", rel_error(dx_num, dx))
    # # print("dprev_h error: ", rel_error(dprev_h_num, dprev_h))
    # # print("dWx error: ", rel_error(dWx_num, dWx))
    # # print("dWh error: ", rel_error(dWh_num, dWh))
    # # print("db error: ", rel_error(db_num, db))
    # #
    # #
    # # N, T, D, H = 2, 3, 4, 5
    # #
    # # x = torch.linspace(-0.1, 0.3, steps=N * T * D, **to_double).view(N, T, D)
    # # h0 = torch.linspace(-0.3, 0.1, steps=N * H, **to_double).view(N, H)
    # # Wx = torch.linspace(-0.2, 0.4, steps=D * H, **to_double).view(D, H)
    # # Wh = torch.linspace(-0.4, 0.1, steps=H * H, **to_double).view(H, H)
    # # b = torch.linspace(-0.7, 0.1, steps=H, **to_double)
    # #
    # # # YOUR_TURN: Implement rnn_forward
    # # h, _ = rnn_forward(x, h0, Wx, Wh, b)
    # # expected_h = torch.tensor(
    # #     [
    # #         [
    # #             [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
    # #             [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
    # #             [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
    # #         ],
    # #         [
    # #             [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
    # #             [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
    # #             [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043],
    # #         ],
    # #     ],
    # #     **to_double,
    # # )
    # # print("h error: ", rel_error(expected_h, h))
    # #
    # #
    # #
    # # reset_seed(0)
    # #
    # # N, D, T, H = 2, 3, 10, 5
    # #
    # # x = torch.randn(N, T, D, **to_double)
    # # h0 = torch.randn(N, H, **to_double)
    # # Wx = torch.randn(D, H, **to_double)
    # # Wh = torch.randn(H, H, **to_double)
    # # b = torch.randn(H, **to_double)
    # #
    # # out, cache = rnn_forward(x, h0, Wx, Wh, b)
    # #
    # # dout = torch.randn(*out.shape, **to_double)
    # #
    # # # YOUR_TURN: Implement rnn_backward
    # # dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
    # #
    # # fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
    # # fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
    # # fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
    # # fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
    # # fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]
    # #
    # # dx_num = compute_numeric_gradient(fx, x, dout)
    # # dh0_num = compute_numeric_gradient(fh0, h0, dout)
    # # dWx_num = compute_numeric_gradient(fWx, Wx, dout)
    # # dWh_num = compute_numeric_gradient(fWh, Wh, dout)
    # # db_num = compute_numeric_gradient(fb, b, dout)
    # #
    # # print("dx error: ", rel_error(dx_num, dx))
    # # print("dh0 error: ", rel_error(dh0_num, dh0))
    # # print("dWx error: ", rel_error(dWx_num, dWx))
    # # print("dWh error: ", rel_error(dWh_num, dWh))
    # # print("db error: ", rel_error(db_num, db))
    # #
    # #
    # #
    # # reset_seed(0)
    # #
    # # N, D, T, H = 2, 3, 10, 5
    # #
    # # # set requires_grad=True
    # # x = torch.randn(N, T, D, **to_double, requires_grad=True)
    # # h0 = torch.randn(N, H, **to_double, requires_grad=True)
    # # Wx = torch.randn(D, H, **to_double, requires_grad=True)
    # # Wh = torch.randn(H, H, **to_double, requires_grad=True)
    # # b = torch.randn(H, **to_double, requires_grad=True)
    # #
    # # out, cache = rnn_forward(x, h0, Wx, Wh, b)
    # #
    # # dout = torch.randn(*out.shape, **to_double)
    # #
    # # # Manual backward:
    # # with torch.no_grad():
    # #     dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
    # #
    # # # Backward with autograd: the magic happens here!
    # # out.backward(dout)
    # #
    # # dx_auto, dh0_auto, dWx_auto, dWh_auto, db_auto = (
    # #     x.grad,
    # #     h0.grad,
    # #     Wx.grad,
    # #     Wh.grad,
    # #     b.grad,
    # # )
    # #
    # # print("dx error: ", rel_error(dx_auto, dx))
    # # print("dh0 error: ", rel_error(dh0_auto, dh0))
    # # print("dWx error: ", rel_error(dWx_auto, dWx))
    # # print("dWh error: ", rel_error(dWh_auto, dWh))
    # # print("db error: ", rel_error(db_auto, db))
    # #
    # #
    # # N, D, T, H = 2, 3, 10, 5
    # #
    # # x = torch.randn(N, T, D, **to_double)
    # # h0 = torch.randn(N, H, **to_double)
    # #
    # # rnn_module = RNN(D, H).to(**to_double)
    # #
    # # # Call forward in module:
    # # hn1 = rnn_module(x, h0)
    # #
    # # # Call without module: (but access weights from module)
    # # # Equivalent to above, we won't do this henceforth.
    # # Wx, Wh, b = rnn_module.Wx, rnn_module.Wh, rnn_module.b
    # # hn2, _ = rnn_forward(x, h0, Wx, Wh, b)
    # #
    # # print("Output error with/without module: ", rel_error(hn1, hn2))
    # #
    # # model = ImageEncoder(pretrained=True, verbose=True).to(device=DEVICE)

    # # N, T, V, D = 2, 4, 5, 3

    # # x = torch.tensor([[0, 3, 1, 2], [2, 1, 0, 3]]).long()
    # # W = torch.linspace(0, 1, steps=V * D, **to_double).view(V, D)

    # # # Copy custom weight vector for sanity check:
    # # model_emb = WordEmbedding(V, D).to(**to_double)
    # # model_emb.W_embed.data.copy_(W)
    # # out = model_emb(x)
    # # expected_out = torch.tensor(
    # #     [
    # #         [
    # #             [0.0, 0.07142857, 0.14285714],
    # #             [0.64285714, 0.71428571, 0.78571429],
    # #             [0.21428571, 0.28571429, 0.35714286],
    # #             [0.42857143, 0.5, 0.57142857],
    # #         ],
    # #         [
    # #             [0.42857143, 0.5, 0.57142857],
    # #             [0.21428571, 0.28571429, 0.35714286],
    # #             [0.0, 0.07142857, 0.14285714],
    # #             [0.64285714, 0.71428571, 0.78571429],
    # #         ],
    # #     ],
    # #     **to_double
    # # )

    # # print("out error: ", rel_error(expected_out, out))
    # #
    # #
    # # def check_loss(N, T, V, p):
    # #     x = 0.001 * torch.randn(N, T, V)
    # #     y = torch.randint(V, size=(N, T))
    # #     mask = torch.rand(N, T)
    # #     y[mask > p] = 0
    # #
    # #     # YOUR_TURN: Implement temporal_softmax_loss
    # #     print(temporal_softmax_loss(x, y, NULL_index).item())
    # #
    # #
    # # check_loss(1000, 1, 10, 1.0)  # Should be about 2.00-2.11
    # # check_loss(1000, 10, 10, 1.0)  # Should be about 20.6-21.0
    # # check_loss(5000, 10, 10, 0.1)  # Should be about 2.00-2.11
    # #
    # #

    # # reset_seed(0)

    # # N, D, W, H = 10, 400, 30, 40
    # # word_to_idx = {"<NULL>": 0, "cat": 2, "dog": 3}
    # # V = len(word_to_idx)
    # # T = 13

    # # model = CaptioningRNN(
    # #     word_to_idx,
    # #     input_dim=D,
    # #     wordvec_dim=W,
    # #     hidden_dim=H,
    # #     cell_type="rnn",
    # #     ignore_index=NULL_index,
    # # )
    # # # Copy parameters for sanity check:
    # # for k, v in model.named_parameters():
    # #     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).view(*v.shape))

    # # images = torch.randn(N, 3, *IMAGE_SHAPE)
    # # captions = (torch.arange(N * T) % V).view(N, T)

    # # loss = model(images, captions)
    # # expected_loss = 150.6090393066

    # # print("loss: ", loss)
    # # print("expected loss: ", expected_loss)
    # # print("difference: ", rel_error(torch.tensor(loss), torch.tensor(expected_loss)))


    # #
    # #
    # #
    # # reset_seed(0)
    # #
    # # # data input
    # # small_num_train = 50
    # # sample_idx = torch.linspace(0, num_train - 1, steps=small_num_train).long()
    # # small_image_data = data_dict["train_images"][sample_idx]
    # # small_caption_data = data_dict["train_captions"][sample_idx]
    # #
    # # # optimization arguments
    # # num_epochs = 80
    # #
    # # # create the image captioning model
    # # model = CaptioningRNN(
    # #     cell_type="rnn",
    # #     word_to_idx=data_dict["vocab"]["token_to_idx"],
    # #     input_dim=400,  # hard-coded, do not modify
    # #     hidden_dim=512,
    # #     wordvec_dim=256,
    # #     ignore_index=NULL_index,
    # # )
    # # model = model.to(**to_float)
    # #
    # # for learning_rate in [1e-3]:
    # #     print("learning rate is: ", learning_rate)
    # #     rnn_overfit, _ = train_captioner(
    # #         model,
    # #         small_image_data,
    # #         small_caption_data,
    # #         num_epochs=num_epochs,
    # #         batch_size=OVR_BATCH_SIZE,
    # #         learning_rate=learning_rate,
    # #         device=DEVICE,
    # #     )
    # #
    # #
    # #
    # # reset_seed(0)
    # #
    # # # data input
    # # small_num_train = num_train
    # # sample_idx = torch.randint(num_train, size=(small_num_train,))
    # # small_image_data = data_dict["train_images"][sample_idx]
    # # small_caption_data = data_dict["train_captions"][sample_idx]
    # #
    # #
    # # # create the image captioning model
    # # rnn_model = CaptioningRNN(
    # #     cell_type="rnn",
    # #     word_to_idx=data_dict["vocab"]["token_to_idx"],
    # #     input_dim=400,  # hard-coded, do not modify
    # #     hidden_dim=512,
    # #     wordvec_dim=256,
    # #     ignore_index=NULL_index,
    # # )
    # #
    # # for learning_rate in [1e-3]:
    # #     print("learning rate is: ", learning_rate)
    # #     rnn_model_submit, rnn_loss_submit = train_captioner(
    # #         rnn_model,
    # #         small_image_data,
    # #         small_caption_data,
    # #         num_epochs=60,
    # #         batch_size=BATCH_SIZE,
    # #         learning_rate=learning_rate,
    # #         device=DEVICE,
    # #     )
    # #
    # #
    # # rnn_model.eval()
    # #
    # # for split in ["train", "val"]:
    # #     sample_idx = torch.randint(
    # #         0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
    # #     )
    # #     sample_images = data_dict[split + "_images"][sample_idx]
    # #     sample_captions = data_dict[split + "_captions"][sample_idx]
    # #
    # #     # decode_captions is loaded from a5_helper.py
    # #     gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])
    # #
    # #     generated_captions = rnn_model.sample(sample_images.to(DEVICE))
    # #     generated_captions = decode_captions(
    # #         generated_captions, data_dict["vocab"]["idx_to_token"]
    # #     )
    # #
    # #     for i in range(VIS_BATCH_SIZE):
    # #         plt.imshow(sample_images[i].permute(1, 2, 0))
    # #         plt.axis("off")
    # #         plt.title(
    # #             f"[{split}] RNN Generated: {generated_captions[i]}\nGT: {gt_captions[i]}"
    # #         )
    # #         plt.show()
    # #
    # #
    # #
    return None



def lstm_test():
    #
    #
    # DEVICE = "cpu"
    # to_float = {"dtype": torch.float32, "device": DEVICE}
    # to_double = {"dtype": torch.float64, "device": DEVICE}
    #
    # # Set a few constants related to data loading.
    # IMAGE_SHAPE = (112, 112)
    # NUM_WORKERS = 1
    #
    # # Batch size used for full training runs:
    # BATCH_SIZE = 256
    #
    # # Batch size used for overfitting sanity checks:
    # OVR_BATCH_SIZE = BATCH_SIZE // 8
    #
    # # Batch size used for visualization:
    # VIS_BATCH_SIZE = 4
    #
    # # Download and load serialized COCO data from coco.pt
    # # It contains a dictionary of
    # # "train_images" - resized training images (IMAGE_SHAPE)
    # # "val_images" - resized validation images (IMAGE_SHAPE)
    # # "train_captions" - tokenized and numericalized training captions
    # # "val_captions" - tokenized and numericalized validation captions
    # # "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"
    #
    # if os.path.isfile("./datasets/coco.pt"):
    #     print("COCO data exists!")
    # else:
    #     print("downloading COCO dataset")
    #
    # # load COCO data from coco.pt, loaf_COCO is implemented in a5_helper.py
    # data_dict = load_coco_captions(path="./datasets/coco.pt")
    #
    # num_train = data_dict["train_images"].size(0)
    # num_val = data_dict["val_images"].size(0)
    #
    # # declare variables for special tokens
    # NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]
    # START_index = data_dict["vocab"]["token_to_idx"]["<START>"]
    # END_index = data_dict["vocab"]["token_to_idx"]["<END>"]
    # UNK_index = data_dict["vocab"]["token_to_idx"]["<UNK>"]
    #
    #
    # # NOTE: step_forward
    # N, D, H = 3, 4, 5
    # x = torch.linspace(-0.4, 1.2, steps=N * D, **to_double).view(N, D)
    # prev_h = torch.linspace(-0.3, 0.7, steps=N * H, **to_double).view(N, H)
    # prev_c = torch.linspace(-0.4, 0.9, steps=N * H, **to_double).view(N, H)
    # Wx = torch.linspace(-2.1, 1.3, steps=4 * D * H, **to_double).view(D, 4 * H)
    # Wh = torch.linspace(-0.7, 2.2, steps=4 * H * H, **to_double).view(H, 4 * H)
    # b = torch.linspace(0.3, 0.7, steps=4 * H, **to_double)
    #
    #
    # # Create module and copy weight tensors for sanity check:
    # model = LSTM(D, H).to(**to_double)
    # model.Wx.data.copy_(Wx)
    # model.Wh.data.copy_(Wh)
    # model.b.data.copy_(b)
    #
    # next_h, next_c = model.step_forward(x, prev_h, prev_c)
    #
    # expected_next_h = torch.tensor(
    #     [
    #         [0.24635157, 0.28610883, 0.32240467, 0.35525807, 0.38474904],
    #         [0.49223563, 0.55611431, 0.61507696, 0.66844003, 0.7159181],
    #         [0.56735664, 0.66310127, 0.74419266, 0.80889665, 0.858299],
    #     ],
    #     **to_double
    # )
    # expected_next_c = torch.tensor(
    #     [
    #         [0.32986176, 0.39145139, 0.451556, 0.51014116, 0.56717407],
    #         [0.66382255, 0.76674007, 0.87195994, 0.97902709, 1.08751345],
    #         [0.74192008, 0.90592151, 1.07717006, 1.25120233, 1.42395676],
    #     ],
    #     **to_double
    # )
    #
    # print("next_h error: ", rel_error(expected_next_h, next_h))
    # print("next_c error: ", rel_error(expected_next_c, next_c))
    #
    #
    # # NOTE: Batch forward
    # N, D, H, T = 2, 5, 4, 3
    # x = torch.linspace(-0.4, 0.6, steps=N * T * D, **to_double).view(N, T, D)
    # h0 = torch.linspace(-0.4, 0.8, steps=N * H, **to_double).view(N, H)
    # Wx = torch.linspace(-0.2, 0.9, steps=4 * D * H, **to_double).view(D, 4 * H)
    # Wh = torch.linspace(-0.3, 0.6, steps=4 * H * H, **to_double).view(H, 4 * H)
    # b = torch.linspace(0.2, 0.7, steps=4 * H, **to_double)
    #
    #
    # # Create module and copy weight tensors for sanity check:
    # model = LSTM(D, H).to(**to_double)
    # model.Wx.data.copy_(Wx)
    # model.Wh.data.copy_(Wh)
    # model.b.data.copy_(b)
    #
    # hn = model(x, h0)
    #
    # expected_hn = torch.tensor(
    #     [
    #         [
    #             [0.01764008, 0.01823233, 0.01882671, 0.0194232],
    #             [0.11287491, 0.12146228, 0.13018446, 0.13902939],
    #             [0.31358768, 0.33338627, 0.35304453, 0.37250975],
    #         ],
    #         [
    #             [0.45767879, 0.4761092, 0.4936887, 0.51041945],
    #             [0.6704845, 0.69350089, 0.71486014, 0.7346449],
    #             [0.81733511, 0.83677871, 0.85403753, 0.86935314],
    #         ],
    #     ],
    #     **to_double
    # )
    #
    # print("hn error: ", rel_error(expected_hn, hn))
    #
    #
    # # NOTE:Captioning Model
    #
    # N, D, W, H = 10, 400, 30, 40
    # word_to_idx = {"<NULL>": 0, "cat": 2, "dog": 3}
    # V = len(word_to_idx)
    # T = 13
    #
    # # YOUR_TURN: Implement CaptioningRNN for lstm
    # model = CaptioningRNN(
    #     word_to_idx,
    #     input_dim=D,
    #     wordvec_dim=W,
    #     hidden_dim=H,
    #     cell_type="lstm",
    #     ignore_index=NULL_index,
    # )
    #
    # model = model.to(DEVICE)
    #
    # for k, v in model.named_parameters():
    #     # print(k, v.shape) # uncomment this to see the weight shape
    #     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).view(*v.shape))
    #
    # images = torch.linspace(
    #     -3.0, 3.0, steps=(N * 3 * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]), **to_float
    # ).view(N, 3, *IMAGE_SHAPE)
    # captions = (torch.arange(N * T) % V).view(N, T)
    #
    # loss = model(images.to(DEVICE), captions.to(DEVICE))
    # expected_loss = torch.tensor(146.3161468505)
    #
    # print("loss: ", loss.item())
    # print("expected loss: ", expected_loss.item())
    # print("difference: ", rel_error(loss, expected_loss))
    #
    #
    # # NOTE: Overfit samll data
    #
    # reset_seed(0)
    #
    # # Data input.
    # small_num_train = 50
    # sample_idx = torch.linspace(0, num_train - 1, steps=small_num_train).long()
    # small_image_data = data_dict["train_images"][sample_idx].to(DEVICE)
    # small_caption_data = data_dict["train_captions"][sample_idx].to(DEVICE)
    #
    # # Create the image captioning model.
    # model = CaptioningRNN(
    #     cell_type="lstm",
    #     word_to_idx=data_dict["vocab"]["token_to_idx"],
    #     input_dim=400,  # hard-coded, do not modify
    #     hidden_dim=512,
    #     wordvec_dim=256,
    #     ignore_index=NULL_index,
    # )
    # model = model.to(DEVICE)
    #
    # for learning_rate in [1e-2]:
    #     print("learning rate is: ", learning_rate)
    #     lstm_overfit, _ = train_captioner(
    #         model,
    #         small_image_data,
    #         small_caption_data,
    #         num_epochs=80,
    #         batch_size=OVR_BATCH_SIZE,
    #         learning_rate=learning_rate,
    #     ) #loss less than 4
    #
    #
    # # NOTE: Captioning sample
    # reset_seed(0)
    #
    # # data input
    # small_num_train = num_train
    # sample_idx = torch.randint(num_train, size=(small_num_train,))
    # small_image_data = data_dict["train_images"][sample_idx]
    # small_caption_data = data_dict["train_captions"][sample_idx]
    #
    # # create the image captioning model
    # lstm_model = CaptioningRNN(
    #     cell_type="lstm",
    #     word_to_idx=data_dict["vocab"]["token_to_idx"],
    #     input_dim=400,  # hard-coded, do not modify
    #     hidden_dim=512,
    #     wordvec_dim=256,
    #     ignore_index=NULL_index,
    # )
    # lstm_model = lstm_model.to(DEVICE)
    #
    # for learning_rate in [1e-3]:
    #     print("learning rate is: ", learning_rate)
    #     lstm_model_submit, lstm_loss_submit = train_captioner(
    #         lstm_model,
    #         small_image_data,
    #         small_caption_data,
    #         num_epochs=60,
    #         batch_size=BATCH_SIZE,
    #         learning_rate=learning_rate,
    #         device=DEVICE,
    #     )
    #
    #
    # lstm_model.eval()
    #
    # for split in ["train", "val"]:
    #     sample_idx = torch.randint(
    #         0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
    #     )
    #     sample_images = data_dict[split + "_images"][sample_idx]
    #     sample_captions = data_dict[split + "_captions"][sample_idx]
    #
    #     # decode_captions is loaded from a5_helper.py
    #     gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])
    #     lstm_model.eval()
    #     generated_captions = lstm_model.sample(sample_images.to(DEVICE))
    #     generated_captions = decode_captions(
    #         generated_captions, data_dict["vocab"]["idx_to_token"]
    #     )
    #
    #     for i in range(VIS_BATCH_SIZE):
    #         plt.imshow(sample_images[i].permute(1, 2, 0))
    #         plt.axis("off")
    #         plt.title(
    #             f"[{split}] LSTM Generated: {generated_captions[i]}\nGT: {gt_captions[i]}"
    #         )
    #         plt.show()
    return None


def attn_lstm_test():
    
DEVICE = "cpu"
to_float = {"dtype": torch.float32, "device": DEVICE}
to_double = {"dtype": torch.float64, "device": DEVICE}

# Set a few constants related to data loading.
IMAGE_SHAPE = (112, 112)
NUM_WORKERS = 1

# Batch size used for full training runs:
BATCH_SIZE = 256

# Batch size used for overfitting sanity checks:
OVR_BATCH_SIZE = BATCH_SIZE // 8

# Batch size used for visualization:
VIS_BATCH_SIZE = 4

# Download and load serialized COCO data from coco.pt
# It contains a dictionary of
# "train_images" - resized training images (IMAGE_SHAPE)
# "val_images" - resized validation images (IMAGE_SHAPE)
# "train_captions" - tokenized and numericalized training captions
# "val_captions" - tokenized and numericalized validation captions
# "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

if os.path.isfile("./datasets/coco.pt"):
    print("COCO data exists!")
else:
    print("downloading COCO dataset")

# load COCO data from coco.pt, loaf_COCO is implemented in a5_helper.py
data_dict = load_coco_captions(path="./datasets/coco.pt")

num_train = data_dict["train_images"].size(0)
num_val = data_dict["val_images"].size(0)

# declare variables for special tokens
NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]
START_index = data_dict["vocab"]["token_to_idx"]["<START>"]
END_index = data_dict["vocab"]["token_to_idx"]["<END>"]
UNK_index = data_dict["vocab"]["token_to_idx"]["<UNK>"]


DEVICE = "cpu"
to_float = {"dtype": torch.float32, "device": DEVICE}
to_double = {"dtype": torch.float64, "device": DEVICE}


N, H = 2, 5
D_a = 4

prev_h = torch.linspace(-0.4, 0.6, steps=N * H, **to_double).view(N, H)
A = torch.linspace(-0.4, 1.8, steps=N * H * D_a * D_a, **to_double).view(N, H, D_a, D_a)

# YOUR_TURN: Implement dot_product_attention
attn, attn_weights = dot_product_attention(prev_h, A)

expected_attn = torch.tensor(
    [
        [-0.29784344, -0.07645979, 0.14492386, 0.36630751, 0.58769115],
        [0.81412643, 1.03551008, 1.25689373, 1.47827738, 1.69966103],
    ],
    **to_double,
)
expected_attn_weights = torch.tensor(
    [
        [
            [0.06511126, 0.06475411, 0.06439892, 0.06404568],
            [0.06369438, 0.06334500, 0.06299754, 0.06265198],
            [0.06230832, 0.06196655, 0.06162665, 0.06128861],
            [0.06095243, 0.06061809, 0.06028559, 0.05995491],
        ],
        [
            [0.05717142, 0.05784357, 0.05852362, 0.05921167],
            [0.05990781, 0.06061213, 0.06132473, 0.06204571],
            [0.06277517, 0.06351320, 0.06425991, 0.06501540],
            [0.06577977, 0.06655312, 0.06733557, 0.06812722],
        ],
    ],
    **to_double,
)

print("attn error: ", rel_error(expected_attn, attn))
print("attn_weights error: ", rel_error(expected_attn_weights, attn_weights))


# NOTE: Test attn_lstm step_forward
N, D, H = 3, 4, 5

x = torch.linspace(-0.4, 1.2, steps=N * D, **to_double).view(N, D)
prev_h = torch.linspace(-0.3, 0.7, steps=N * H, **to_double).view(N, H)
prev_c = torch.linspace(-0.4, 0.9, steps=N * H, **to_double).view(N, H)
attn = torch.linspace(0.6, 1.8, steps=N * H, **to_double).view(N, H)

Wx = torch.linspace(-2.1, 1.3, steps=4 * D * H, **to_double).view(D, 4 * H)
Wh = torch.linspace(-0.7, 2.2, steps=4 * H * H, **to_double).view(H, 4 * H)
b = torch.linspace(0.3, 0.7, steps=4 * H, **to_double)
Wattn = torch.linspace(1.3, 4.2, steps=4 * H * H, **to_double).view(H, 4 * H)

# Create module and copy weight tensors for sanity check:
model = AttentionLSTM(D, H).to(**to_double)
model.Wx.data.copy_(Wx)
model.Wh.data.copy_(Wh)
model.b.data.copy_(b)
model.Wattn.data.copy_(Wattn)

next_h, next_c = model.step_forward(x, prev_h, prev_c, attn)


expected_next_h = torch.tensor(
    [
        [0.53704256, 0.59980774, 0.65596820, 0.70569729, 0.74932626],
        [0.78729857, 0.82010653, 0.84828362, 0.87235677, 0.89283167],
        [0.91017981, 0.92483119, 0.93717126, 0.94754073, 0.95623746],
    ],
    **to_double
)
expected_next_c = torch.tensor(
    [
        [0.59999328, 0.69285041, 0.78570758, 0.87856479, 0.97142202],
        [1.06428558, 1.15714276, 1.24999992, 1.34285708, 1.43571424],
        [1.52857143, 1.62142857, 1.71428571, 1.80714286, 1.90000000],
    ],
    **to_double
)

print("next_h error: ", rel_error(expected_next_h, next_h))
print("next_c error: ", rel_error(expected_next_c, next_c))


# NOTE:forward
N, D, H, T = 2, 5, 4, 3
D_a = 4

x = torch.linspace(-0.4, 0.6, steps=N * T * D, **to_double).view(N, T, D)
A = torch.linspace(-0.4, 1.8, steps=N * H * D_a * D_a, **to_double).view(
    N, H, D_a, D_a
)

Wx = torch.linspace(-0.2, 0.9, steps=4 * D * H, **to_double).view(D, 4 * H)
Wh = torch.linspace(-0.3, 0.6, steps=4 * H * H, **to_double).view(H, 4 * H)
Wattn = torch.linspace(1.3, 4.2, steps=4 * H * H, **to_double).view(H, 4 * H)
b = torch.linspace(0.2, 0.7, steps=4 * H, **to_double)


# Create module and copy weight tensors for sanity check:
model = AttentionLSTM(D, H).to(**to_double)
model.Wx.data.copy_(Wx)
model.Wh.data.copy_(Wh)
model.b.data.copy_(b)
model.Wattn.data.copy_(Wattn)

# YOUR_TURN: Implement attention_forward
hn = model(x, A)

expected_hn = torch.tensor(
    [
        [
            [0.56141729, 0.70274849, 0.80000386, 0.86349400],
            [0.89556391, 0.92856726, 0.94950579, 0.96281018],
            [0.96792077, 0.97535465, 0.98039623, 0.98392994],
        ],
        [
            [0.95065880, 0.97135490, 0.98344373, 0.99045552],
            [0.99317679, 0.99607466, 0.99774317, 0.99870293],
            [0.99907382, 0.99946784, 0.99969426, 0.99982435],
        ],
    ],
    **to_double
)

print("h error: ", rel_error(expected_hn, hn))



reset_seed(0)

N, D, W, H = 10, 400, 30, 40
word_to_idx = {"<NULL>": 0, "cat": 2, "dog": 3}
V = len(word_to_idx)
T = 13

# YOUR_TURN: Modify CaptioningRNN for attention
model = CaptioningRNN(
    word_to_idx,
    input_dim=D,
    wordvec_dim=W,
    hidden_dim=H,
    cell_type="attn",
    ignore_index=NULL_index,
)
model = model.to(DEVICE)

for k, v in model.named_parameters():
    # print(k, v.shape) # uncomment this to see the weight shape
    v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).view(*v.shape))

images = torch.linspace(
    -3.0, 3.0, steps=(N * 3 * IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
).view(N, 3, *IMAGE_SHAPE)
captions = (torch.arange(N * T) % V).view(N, T)

loss = model(images.to(DEVICE), captions.to(DEVICE))
expected_loss = torch.tensor(8.0156393051)

print("loss: ", loss.item())
print("expected loss: ", expected_loss.item())
print("difference: ", rel_error(loss, expected_loss))


#
# reset_seed(0)
#
# # data input
# small_num_train = 50
# sample_idx = torch.linspace(0, num_train - 1, steps=small_num_train).long()
# small_image_data = data_dict["train_images"][sample_idx]
# small_caption_data = data_dict["train_captions"][sample_idx]
#
# # create the image captioning model
# model = CaptioningRNN(
#     cell_type="attn",
#     word_to_idx=data_dict["vocab"]["token_to_idx"],
#     input_dim=400,  # hard-coded, do not modify
#     hidden_dim=512,
#     wordvec_dim=256,
#     ignore_index=NULL_index,
# )
#
#
# for learning_rate in [1e-3]:
#     print("learning rate is: ", learning_rate)
#     attn_overfit, _ = train_captioner(
#         model,
#         small_image_data,
#         small_caption_data,
#         num_epochs=80,
#         batch_size=OVR_BATCH_SIZE,
#         learning_rate=learning_rate,
#         device=DEVICE,
#     )
#

reset_seed(0)

# data input
small_num_train = num_train
sample_idx = torch.randint(num_train, size=(small_num_train,))
small_image_data = data_dict["train_images"][sample_idx]
small_caption_data = data_dict["train_captions"][sample_idx]

# create the image captioning model
attn_model = CaptioningRNN(
    cell_type="attn",
    word_to_idx=data_dict["vocab"]["token_to_idx"],
    input_dim=400,  # hard-coded, do not modify
    hidden_dim=512,
    wordvec_dim=256,
    ignore_index=NULL_index,
)
attn_model = attn_model.to(DEVICE)

for learning_rate in [1e-3]:
    print("learning rate is: ", learning_rate)
    attn_model_submit, attn_loss_submit = train_captioner(
        attn_model,
        small_image_data,
        small_caption_data,
        num_epochs=60,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        device=DEVICE,
    )


# Sample a minibatch and show the reshaped 112x112 images,
# GT captions, and generated captions by your model.

for split in ["train", "val"]:
    sample_idx = torch.randint(
        0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
    )
    sample_images = data_dict[split + "_images"][sample_idx]
    sample_captions = data_dict[split + "_captions"][sample_idx]

    # decode_captions is loaded from a5_helper.py
    gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])
    attn_model.eval()
    generated_captions, attn_weights_all = attn_model.sample(sample_images.to(DEVICE))
    generated_captions = decode_captions(
        generated_captions, data_dict["vocab"]["idx_to_token"]
    )

    for i in range(VIS_BATCH_SIZE):
        plt.imshow(sample_images[i].permute(1, 2, 0))
        plt.axis("off")
        plt.title(
            "%s\nAttention LSTM Generated:%s\nGT:%s"
            % (split, generated_captions[i], gt_captions[i])
        )
        plt.show()

        tokens = generated_captions[i].split(" ")

        vis_attn = []
        for j in range(len(tokens)):
            img = sample_images[i]
            attn_weights = attn_weights_all[i][j]
            token = tokens[j]
            img_copy = attention_visualizer(img, attn_weights, token)
            vis_attn.append(transforms.ToTensor()(img_copy))

        plt.rcParams["figure.figsize"] = (20.0, 20.0)
        vis_attn = make_grid(vis_attn, nrow=8)
        plt.imshow(torch.flip(vis_attn, dims=(0,)).permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        plt.rcParams["figure.figsize"] = (10.0, 8.0)
        

    return None