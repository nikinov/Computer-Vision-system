#
#
#
#
# Nicholas Novelle July 2021
#

import matplotlib.pyplot as plt

def print_metrix(e, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc):
    print('epoch :', (e + 1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc))
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc))


def plot_metrix(running_loss_history, val_running_loss_history, running_corrects_history, val_running_corrects_history):
    plt.clf()
    plt.plot(running_loss_history, label='training loss')
    plt.plot(val_running_loss_history, label='validation loss')
    plt.legend()
    plt.savefig('image_loss.png', dpi=90, bbox_inches='tight')

    plt.clf()
    plt.plot(running_corrects_history, label='training accuracy')
    plt.plot(val_running_corrects_history, label='validation accuracy')
    plt.legend()
    plt.savefig('image_acc.png', dpi=90, bbox_inches='tight')