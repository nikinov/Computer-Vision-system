from PIL import Image
import matplotlib.pyplot as plt

from DataSampler import DataSampler


def main():

    dataloader = DataSampler(
        path=r'..\..\Assets5082',
        labels={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'E': 10},
        verbose=False
    )
	
	# create splits
    dataloader.create_splits(shuffle=True, equalize=True)
    dataloader.save('list.csv')

	# load from file
    dataloader.clear()
    dataloader.load('list.csv')

    for label, fn in dataloader.get_label_filename('trn'):
        print(f'{label}: {fn}')

    for label, img in dataloader.get_label_image('trn'):
        plt.clf()
        plt.imshow(img)
        plt.show()
        break

if __name__ == "__main__":
    main()
    print('done')
