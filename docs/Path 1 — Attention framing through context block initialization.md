Utilize features like eye-gaze to bias the attention on an explicit area of interest

- introducing features like eye-gaze to enhance the prediction by defining an ==area of interest==
- to achieve this, we tried to modify ==context block== initialization logic from random to biased with respect to a weighted map generated based on the ==area of interest==
- **-** it is still shady for me whether I’ve injected this mask as a temporal mask or not

![[image.png]]