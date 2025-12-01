"""
Real-time Face Recognition with Age, Gender, and Emotion Detection
웹캠을 통한 실시간 얼굴 인식 및 나이, 성별, 감정 분석 (GUI 버전)

메인 실행 파일
Main Entry Point
"""

from ui import App


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # X 버튼 눌렀을 때
    app.mainloop()
