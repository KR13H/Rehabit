# Rehabit
A blend of rehab + habit -> emphasis of daily recovery habits, consistency and progress. 
WHAT IS IT?
Rehabit is a simple, real-time pose tracking tool built for rehabilitation exercises. It uses MediaPipe and OpenCV to detect body keypoints, then analyzes your movements to give instant feedback on form and symmetry. This helps people recovering from stroke, paralysis, or other mobility challenges.
What We’ve Built
A Flask + SocketIO server that streams webcam frames to the backend
A MediaPipe engine that finds 33 body landmarks each frame
Exercise trackers for arm raises, sit-to-stand, and marching in place
Real-time visual overlays and audio/text cues to correct form
Session saving so you can review progress over time