# EnhancedLaneVision

<img src="images/10.jpg" alt="drawing" width="400"/>

## Project Goal
This project leverages the tools in Python and OpenCV to detect lanes in dash cam footage. My goal was to become more acquaintted with AI and computer vision algorithms to in my software to identify straight lanes, providing real-time visual feedback to users.

---


## What was implemented
In computer vision, a common tool for isolating objects in an image comes in the form of *masking*. To mask an image, we need to create a completely black (blank) image with the same height and width as the image we intend to mask. When we place a white polygon, such as a triangle or a square, within the blank image, we can merge this with the original, resulting in a new image containing contents of the original image only within the boundaries of the polygon.

For dash cam frames, I went with a simple removal of the upper 1/3rd of the image since that is where most of the background is.

**Pre Processing**
To get started, I first needed to find a good mask to identify where the road was and cut out the background parts of the image. Most of the roads in the dashcam footage took up a trapezoidal area within the frame and I wanted to mask just this trapezoid. So I used the first 8 frames of a given dashcam video as a way of priming my algorithm that would identify the images in real time.

### *Color Thresholding*
A more advanced way of masking is through use of color thresholds. Given a range of colors, the 'cv2.inRange()' function can be used to essentially mask the image by color. With this in mind, I used a color threshold to capture a range of yellows and whites, matching the lane colors that were commonly found in dash cam footage.
This would cut out large portions of the picture, leaving behind the lanes along with a few remaining artifacts.

### *Sobel Edge Detection*
There are many ways of detecting edges, borders, or lines of an image in computer vision. One such was is sobel edge detection, which can be good for identifying linear patterns either in the vertical direction (cv2.Sobel(dx=1, dy=0)) or the horizontal directon (cv2.Sobel(dx=0, dy=1)). Since most lanes in dashcam footage are usually oriented with steep curves, I used sobel in the x direction to identify prominent vertically oriented edges within the image.

### *arbitrary masking for a region of interest*
A small triangular polygon was used tomask a portion of the image. It was intended for one specific video, but ended up working for a lot of the data I tried. So I just kept it

### *Hough Line Transform for Lane Detection*
The Hough Line Transform is a pivotal technique in this project, effectively utilized for discerning linear patterns in an image. Initially, the algorithm attempts to identify potential lines in the frame. To refine this, a filtering process is employed, prioritizing lines closer to the image's center, as they are more likely to represent lane markings. Subsequently, an evaluation of the average slope intercept of these lines is conducted. This step is crucial as it segregates the detected lines based on their slope orientation: positive slopes indicating left-side lanes and negative slopes for right-side lanes. 
By averaging the lines with similar slopes, I derived two distinct sets of averaged lines, each representing the respective lane markings. Utilizing the first 8 frames of the dashcam footage, an average of these optimal left and right line pairs is calculated. This average serves as a foundation for hypothesizing the lane positions.
In the next phase, a trapezoidal region, approximating the road area, is constructed around these average lines. This trapezoid then facilitates the creation of smaller, more targeted masks. By isolating the trapezoid's center and bifurcating it into left and right segments, the algorithm enhances its precision in differentiating between the lanes. This approach mitigates potential confusion where elements of one lane might be misconstrued as part of the other, thereby bolstering the accuracy of lane identification.

### *Intended Next Steps*
Initially, my plan involved further refining the pre-processing mask to reduce screen artifacts. This refinement was to be achieved through two primary strategies:
1. **Contour Detection:** Modifying the trapezoidal mask to more accurately encompass the lane areas while excluding irrelevant sections of the frame.
2. **Color Threshold Adjustments:** Based on the contours detected, fine-tuning the color thresholding ranges for yellow and white to enhance the precision of lane identification and reduce noise from non-lane elements.

However, this intended course of action was altered as the project progressed, leading to a different approach in the subsequent steps.

### *Actual Next Steps and Innovations*
In practice, the mask derived from the Hough Line Transform often proved sufficient for lane detection, though some artifacts persisted. To address this, I implemented an additional algorithm focusing on line pattern recognition within the trapezoid-shaped lane area. This algorithm specifically aimed to identify the most prominent dotted (largest hough line with several spacings) and solid lines (largest linearly oriented rectangular contour) on either side of the trapezoid.

The challenge arose when distinguishing actual lane markings from similar-looking artifacts. Despite these occasional confusions, the method generally yielded reliable results. A key observation was that the largest dotted and solid lines identified often coincided with actual lane lines, particularly when these lines were consistent across frames, indicating a lane marking irrespective of it being solid or dotted.

This discovery led to a novel approach: recording instances where the largest solid and dotted lines converged into a single line. This convergence served as a reliable indicator of actual lane presence. Thus, the algorithm not only identified lanes but also guided the slope calculation for real-time lane tracking, effectively differentiating between genuine lane lines and artifacts. This approach significantly improved the system's capability to process and filter out irrelevant lines, enhancing overall accuracy in real-time lane detection.

### *Real time processing*
Some things are the same, but less complicated as I used the trapezoid mask along with the slopes I found, as 'heuristic' guides for identifying the lanes. The order of the real time algorithm was shorter:
- Color threshold the image
- apply a simple binary threshold on that image (not edge detection, but setting a color threshold, making any color above the threshold value white, any color below it black)
- apply the trapezoid mask found from pre-processing
- find the hough lines, then filter out the ones that were way off from the average slope
    - some times the hough lines came back with nothing, as my parameters were a little more strict here than before (i.e. I made the requirement to look for longer lines, before I was kind of loose on how long the line I was looking for needed to be)
    - I'd change the average slope along with the average slope of every 80th frame (just to shake it up a lil bit)
    - when the Hough Line identifier came back with zilch, I'd just display the average line (not the best, would've been better to do a variable line length, but ehh...)
- display those lines found onto the original image.

---

## Challenges faced along the way
### *Line and HSV color handling in openCV*
One tricky thing was the way values such as HSV colors and lines are formatted in OpenCV. there is no dedicated color object or line object for these values, but instead, each value is represented in a numpy array. so a line would be an array of length four [x1, y1, x2,y2]. RGB values were simple enough [0-255,0-255,0-255], but it got trickier when handling HSV values. 
The HSV color wheel is typically represented with a cone. The Hue of a color represented in degrees(0-360), while the Saturation and Value of the color are represented as percentages (0%-100%). In openCV, the HSV values were represented by dividing the hue in half, and using 255 as 100% so HSV values would look like: [0-180, 0-255, 0-255].


### *Knowing which Algorithm would work*
I mentioned before that I used a rectangle to cut out the upper 1/3rd rectangle as well as a triangle near the center the image as a simple way of ridding the background. The more advanced alternative would have been using a combination of thresholds along with a distance transform to segment the background from the foreground. 
Distance transforms are useful as you can use this to determine the distance between pixel clusters. One thing I experimented with was cutting out white pixel clusters depending on some distance they had to black pixels, with the intention of cutting out larger bulkier artifacts. but this had limited use, would at times cut out parts of the lanes, and generally leave behind the edges of the bulky object (meaning more ways to confuse something non linear for a line!)
I may try and implement this later such that the lowest point of the background could be used to find the horizon in the image, then cut out a trapezoid relative to that horizon.It would be cleaner 
Another thing I tried was the watershed algorithm. This was a pretty interesting way of segmenting the image. You can use it to identify contours of 
I was originally thinking of using this to detect the road vs non-road foreground and background. Didn't quite work, as it made a lot of partitions. 
Again, may work, but I didn't experiment enough.

---

## What I might try in the future:
- Succesfully implementing the watershed algorithm (to remove non-lane object image artifacts)
- Succesfully implementing the distance transform (in tandem with watershed)
- Application of perspective transform (this was used by a lot of other lane detection projects I saw)
- Maybe trying camera calibration and limiting distortion of the camera image. (this was also used by a lot of other lane detection projects I saw)
