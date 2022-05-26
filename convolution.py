import numpy as np
import matplotlib.pyplot as plt


class Convolution:

    def __init__(self, image, filter):
        """
        Input
        :param image: input image
        :param filter: input filter
        """
        self.image = np.array(image)
        self.filter = np.array(filter)
        self.blurry_image = []

    def print_image(self, blurry=False):
        """
        Print the image pixel intensity matrix
        :param blurry: If True, we print the intensity of the blurry image,
                       otherwise, we print the intensity of the original image.
        :return: M times 1 * N image pixel intensity vector. M is the number of the image rows,
                 and N is the number of the image columns
        """
        if blurry:
            image_print = self.blurry_image
        else:
            image_print = self.image

        for row in image_print:
            print(row)

    def visualization(self, blurry=False):
        """
        Input image visualization
        :param blurry: If True, we visualize the blurry image in the grayscale,
                             otherwise, we visualize the original image in the grayscale.
        :return: None
        """
        if blurry:
            plt.imshow(self.blurry_image, cmap='gray')
        else:
            plt.imshow(self.image, cmap='gray')
        plt.show()

    def zero_pad(self):
        """
        Zero pad the image
        :return: Padded image
        """
        filtersize = len(self.filter)
        padded_image = np.pad(self.image, pad_width=filtersize // 2)
        return padded_image

    def apply_filter(self):
        """
        apply the Convolutional Kernel to the image
        :return: the blurred image
        """
        filtersize = len(self.filter)
        image_height, image_width = len(self.image), len(self.image[0])
        padded_image = self.zero_pad()

        self.blurry_image = []
        for i in range(image_height):
            for j in range(image_width):
                new_intensity = np.sum(padded_image[i:filtersize + i, j:filtersize + j] * self.filter)
                self.blurry_image.append(new_intensity)

        self.blurry_image = np.array(self.blurry_image).reshape((image_height, image_width))
        return self.blurry_image


if __name__ == '__main__':
    plt.ion()

    # gaussian blur filter
    filter = [
        [1 / 16, 1 / 8, 1 / 16],
        [1 / 8, 1 / 4, 1 / 8],
        [1 / 16, 1 / 8, 1 / 16],
    ]

    # 9x11 single channel image of intensities
    image = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
    ]

    # ensure the blurred_image matches the expected_image
    expected_image = [
        [0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 4.875],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.5],
        [0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0, 4.875]
    ]

    a = Convolution(image, filter)

    # apply the filter to the image
    blurred_image = a.apply_filter()

    # image visualization
    a.visualization(blurry=False)

    # print the pixel intensity
    a.print_image(blurry=False)

    # check if the blurred image is correct
    print("blurred_image == expected_image?", blurred_image == expected_image)

    plt.ioff()
    plt.show()
