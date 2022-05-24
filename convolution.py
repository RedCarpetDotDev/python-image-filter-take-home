# gaussian blur filter
filter = [
  [1/16, 1/8, 1/16],
  [1/8, 1/4, 1/8],
  [1/16, 1/8, 1/16],
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

def print_image(image):
  for row in image:
    print(row)  

# apply `filter` to `image`
# return the blurred image
def apply_filter(image, filter):
  # pad the input image with zeroes
  # start with an 2d array of zeroes that can accommodate the embedded image
  rows = len(image)
  cols = len(image[0])
  pad_rows = len(filter) // 2
  pad_cols = len(filter[0]) // 2
  padded_image = [
    [0 for i in range(cols + pad_rows * 2)]
     for j in range(rows + pad_cols * 2)
  ]

  # fill the interior with our image
  for i in range(pad_rows, rows + pad_rows):
    for j in range(pad_cols, cols + pad_cols):
      padded_image[i][j] = image[i - 1][j - 1]

  # create an output image, the same size as the input
  output = [
    [0 for i in range(cols)]
     for j in range(rows)
  ]
  for i in range(0, rows):
    for j in range(0, cols):
      
      # apply the patch to the appropriate pixel
      sum = 0
      for y in range(0, len(filter)):
        for x in range(0, len(filter[0])):
          # translate to the appropriate position within the padded image 
          image_i = i + y
          image_j = j + x
          sum += padded_image[image_i][image_j] * filter[y][x]
      output[i][j] = sum
      
  return output

# apply the filter to the image
blurred_image = apply_filter(image, filter)

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

print("blurred_image == expected_image?", blurred_image == expected_image)
