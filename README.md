# Rehabit  
A blend of Rehab + Habit → Empowering consistent recovery through AI.

---

## Overview
Rehabit is an AI-powered rehabilitation assistant built to help stroke recovery and mobility-challenged patients perform guided physical therapy exercises at home.

By combining pose estimation, AI feedback, and voice coaching, Rehabit turns rehabilitation into a consistent daily habit — focusing on progress, balance, and confidence.

---

## Problem
Stroke rehabilitation requires repetitive, supervised exercises — but most patients lack daily guidance or immediate feedback when practicing at home.  
Rehabit solves this by providing:
- AI-driven real-time posture correction  
- Voice and text feedback for form and symmetry  
- Progress tracking and motivation through scores

---

## Tech Stack
| Layer | Tools Used |
|-------|-------------|
| Programming Language | Python |
| AI / ML | MediaPipe (Pose Detection), OpenCV (Image Processing) |
| Web Framework | Flask + SocketIO (Real-time streaming) |
| AI Agents | Gemini (Form feedback and analysis), ElevenLabs (Voice guidance) |
| Frontend | HTML, CSS, JavaScript |
| Data | JSON-based angle and symmetry analysis |

---

## Core Features
- AI-powered pose tracking using MediaPipe to detect 33 body landmarks  
- Real-time feedback analyzing body symmetry, range of motion, and form  
- Voice and text guidance using Gemini and ElevenLabs  
- Exercise modules including Arm Raises, Elbow Flexion, Sit-to-Stand, and Marching in Place  
- Live overlays that show posture and movement cues  
- Session reports to track improvement and accuracy over time  
- Scoring system for performance evaluation  

---

## System Architecture
1. Frontend  
   - Captures webcam input  
   - Displays pose overlay and exercise guide  
   - Streams video frames to backend in real time  

2. Backend (Flask + SocketIO)  
   - Processes frames using MediaPipe and OpenCV  
   - Computes joint angles, range of motion, and symmetry values  
   - Passes data to AI agent (Gemini) for analysis  

3. AI Agents  
   - Gemini interprets form accuracy and provides exercise advice  
   - ElevenLabs converts AI feedback to natural voice for user motivation  

---

## Supported Exercises
Each exercise includes posture detection, form validation, and AI-generated audio coaching.

| Exercise | Focus Area | AI Feedback |
|-----------|-------------|-------------|
| Arm Raises | Shoulder mobility | Detects uneven elevation and posture tilt |
| Elbow Flexion | Arm coordination | Tracks bend angle and smoothness |
| Sit-to-Stand | Leg strength and balance | Ensures full motion range |
| March in Place | Lower limb coordination | Monitors cadence and knee height |

---

## Installation and Setup

1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/Rehabit.git
   cd Rehabit
