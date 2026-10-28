APP_NAME = "Multimodal Attendance & Engagement Agent"

AUDIO_LABELS = ["Silence", "Instructor Lecture", "Student Discussion", "High Ambient Distraction"]

AUDIO_COLORS = {
    "Silence": "#94A3B8",
    "Instructor Lecture": "#3B82F6",
    "Student Discussion": "#10B981",
    "High Ambient Distraction": "#EF4444",
}

# How much each acoustic state contributes to engagement (0-1).
AUDIO_SCORE = {
    "Silence": 0.25,
    "Instructor Lecture": 0.60,
    "Student Discussion": 1.00,
    "High Ambient Distraction": 0.0,
}
