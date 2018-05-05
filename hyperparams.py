class Hyperparams:
    '''Hyperparameters'''
    # train Scheme
    batch_size = 128
    num_epochs = 200

    # model details
    embed_size = 100
    tau_value = 1.0
    vocab_size = 50000
    # M: number of codebooks (subcodes)
    M = 32
    # K: number of vectors in each codebook
    K = 16

    # data
    np_embed_path = 'datas/glove.6B.100d.npy'

    # model
    modelname = 'model'
    is_earlystopping = False
