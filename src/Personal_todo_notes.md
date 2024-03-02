# TODO: 
- Keep track of centroid and use the new lines to inform them of where a new centroid and new mask should be
- try removing the line and hsv classes at some point
- so you'll need to apply some linear algebra. I'm thinking, we look for:
  - I'm going to say the eigenvalue (i know that's not the right word) but
    the eigenvalue to multiply our two linear vectors by such that they become nearly parallel in that vector space. 

# NOTE: 
- Gradients are apparently very faulty

Note that I'm using OpenCV and python, no use of Machine Learning.

Data: 4 different dash cam videos
Preprocessing algorithm: gets initial 8 frames and iterates through them by:
 1. identify where the background is (cv2.morphologyEx, cv2.dilate), remove a rectangular region where that background is
 2. Apply color thresholding onto the image (white and yellow only remain)
 3. apply sobel edge detection (in the x-direction)
 4. isolate a trapezoid in the image (bottom is the height, top is where the background was cutoff)
 5. identify HoughLines, remove the outliers.
 6. find an average slope intercept
 7. determine generally where the averaged lines were
 8. isolate a smaller trapezoid based on those lines
 9. create two new half trapezoid masks to find left and right lanes in isolate for future frames.

