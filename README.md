# Image-stiching
 Exercise of Panorama Preparation Functions. Combines selecting the appropriate points manually and implementing a homographic matrix to translate the image

# Requirements
- python 3.7 (tested)
 - numpy library
 - scipy library
 - cv2 library (just for the matching point, all the stiching function implemented from scratch)
 - matplotlib library

# Walk through
1. Take two consecutive photos.
2. Name the first photo src.jpg and the second dst.jpg
3. Run the manually matching function first, and determine 50 (configurable) matching points between the pictures.
4. Then press some key, to save the matches matrix as match.mat
5. Now you can run the stiching function.

Enjoy,
Sunny
