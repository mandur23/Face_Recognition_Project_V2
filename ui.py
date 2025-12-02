"""
UI 애플리케이션 모듈
GUI Application Module
"""

import customtkinter as ctk
import cv2
import threading
from PIL import Image, ImageTk
import time
from typing import Optional
from face_analyzer import FaceAnalyzer


class App(ctk.CTk):
    """메인 UI 애플리케이션 클래스"""

    def __init__(self):
        super().__init__()

        # 윈도우 설정
        self.title("AI Face Analysis Dashboard")
        self.geometry("1200x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # 그리드 레이아웃 (3단 구성: 사이드바 | 비디오 | 정보패널)
        self.grid_columnconfigure(1, weight=3)  # 비디오 영역
        self.grid_columnconfigure(2, weight=1)  # 정보 패널 영역
        self.grid_rowconfigure(0, weight=1)

        # 좌측 사이드바
        self._setup_sidebar()

        # 중앙 비디오 프레임
        self._setup_video_frame()

        # 우측 분석 정보 패널
        self._setup_info_panel()

        # 변수 초기화
        self.cap = None
        self.analyzer = FaceAnalyzer(analysis_interval=15, loading_callback=self.update_loading_status)
        self.is_running = False
        self.prev_time = 0
        self.is_camera_loading = False

    def _on_brightness_change(self, value):
        """밝기 슬라이더 변경 시 호출"""
        self.brightness_value_label.configure(text=f"{int(value)}")

    def _on_contrast_change(self, value):
        """대비 슬라이더 변경 시 호출"""
        self.contrast_value_label.configure(text=f"{int(value)}")

    def _apply_camera_effects(self, frame):
        """카메라 효과 적용"""
        # 좌우 반전
        if self.flip_horizontal_var.get():
            frame = cv2.flip(frame, 1)

        # 흑백 모드
        if self.grayscale_var.get():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 밝기 조절
        brightness = self.brightness_var.get()
        if brightness != 0:
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)

        # 대비 조절
        contrast = self.contrast_var.get()
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)

        return frame

    def _setup_sidebar(self):
        """사이드바 설정"""
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        # 로고 및 버전
        self.logo = ctk.CTkLabel(
            self.sidebar_frame,
            text="VISION AI",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.ver_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="v2.0 Dashboard",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        )
        self.ver_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # 컨트롤 버튼
        self.start_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="▶ 카메라 시작",
            command=self.start_camera,
            fg_color="#2CC985",
            hover_color="#229A66",
            text_color="black"
        )
        self.start_btn.grid(row=2, column=0, padx=20, pady=10)

        self.stop_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="⏹ 카메라 정지",
            command=self.stop_camera,
            fg_color="#FF4B4B",
            hover_color="#CC3333"
        )
        self.stop_btn.grid(row=3, column=0, padx=20, pady=10)

        # 구분선
        ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30").grid(
            row=4, column=0, sticky="ew", padx=20, pady=20
        )

        # 옵션 스위치
        self.show_overlay_var = ctk.BooleanVar(value=True)
        self.overlay_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="얼굴 박스 표시",
            variable=self.show_overlay_var
        )
        self.overlay_switch.grid(row=5, column=0, padx=20, pady=10, sticky="w")

        # 구분선
        ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30").grid(
            row=6, column=0, sticky="ew", padx=20, pady=20
        )

        # 카메라 설정 섹션
        ctk.CTkLabel(
            self.sidebar_frame,
            text="카메라 설정",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=7, column=0, padx=20, pady=(0, 10), sticky="w")

        # 좌우 반전
        self.flip_horizontal_var = ctk.BooleanVar(value=True)
        self.flip_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="좌우 반전",
            variable=self.flip_horizontal_var
        )
        self.flip_switch.grid(row=8, column=0, padx=20, pady=5, sticky="w")

        # 흑백 모드
        self.grayscale_var = ctk.BooleanVar(value=False)
        self.grayscale_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="흑백 모드",
            variable=self.grayscale_var
        )
        self.grayscale_switch.grid(row=9, column=0, padx=20, pady=5, sticky="w")

        # 밝기 조절
        ctk.CTkLabel(
            self.sidebar_frame,
            text="밝기",
            font=ctk.CTkFont(size=12)
        ).grid(row=10, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.brightness_var = ctk.DoubleVar(value=0.0)  # -100 ~ 100
        self.brightness_slider = ctk.CTkSlider(
            self.sidebar_frame,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self._on_brightness_change
        )
        self.brightness_slider.grid(row=11, column=0, padx=20, pady=5, sticky="ew")
        self.brightness_value_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="0",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.brightness_value_label.grid(row=12, column=0, padx=20, pady=(0, 5))

        # 대비 조절
        ctk.CTkLabel(
            self.sidebar_frame,
            text="대비",
            font=ctk.CTkFont(size=12)
        ).grid(row=13, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.contrast_var = ctk.DoubleVar(value=0.0)  # -100 ~ 100
        self.contrast_slider = ctk.CTkSlider(
            self.sidebar_frame,
            from_=-100,
            to=100,
            variable=self.contrast_var,
            command=self._on_contrast_change
        )
        self.contrast_slider.grid(row=14, column=0, padx=20, pady=5, sticky="ew")
        self.contrast_value_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="0",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.contrast_value_label.grid(row=15, column=0, padx=20, pady=(0, 10))

        # 그리드 행 조정
        self.sidebar_frame.grid_rowconfigure(16, weight=1)

        # 상태 라벨
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="System Ready",
            text_color="gray"
        )
        self.status_label.grid(row=17, column=0, padx=20, pady=20)

    def _setup_video_frame(self):
        """비디오 프레임 설정"""
        self.video_container = ctk.CTkFrame(self, fg_color="black")
        self.video_container.grid(row=0, column=1, sticky="nsew", padx=(10, 5), pady=10)

        self.video_label = ctk.CTkLabel(self.video_container, text="", cursor="cross")
        self.video_label.pack(expand=True, fill="both", padx=2, pady=2)

        # 로딩 표시
        self.loading_label = ctk.CTkLabel(
            self.video_container,
            text="",
            font=ctk.CTkFont(size=20),
            text_color="#2CC985"
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")

    def _setup_info_panel(self):
        """정보 패널 설정"""
        self.info_frame = ctk.CTkFrame(self, width=250, corner_radius=10)
        self.info_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        self.info_frame.grid_columnconfigure(0, weight=1)

        # 제목
        ctk.CTkLabel(
            self.info_frame,
            text="Real-time Analysis",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=20)

        # 얼굴 수 표시
        self.card_faces = self.create_info_card(
            self.info_frame, "감지된 얼굴 수", "0 명", "👥"
        )

        # 스크롤 가능한 얼굴 정보 영역
        self.faces_scroll_frame = ctk.CTkScrollableFrame(self.info_frame, fg_color="transparent")
        self.faces_scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 얼굴 카드들을 저장할 딕셔너리
        self.face_cards = {}

        # FPS 표시
        self.fps_label = ctk.CTkLabel(
            self.info_frame,
            text="FPS: 00",
            font=ctk.CTkFont(family="Consolas", size=14),
            text_color="gray"
        )
        self.fps_label.pack(side="bottom", pady=20)

    def create_info_card(self, parent, title, value_text, icon):
        """재사용 가능한 정보 카드 생성"""
        card = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=8)
        card.pack(fill="x", padx=15, pady=10)

        # 타이틀
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(anchor="w", padx=15, pady=(10, 0))

        # 값과 아이콘 프레임
        val_frame = ctk.CTkFrame(card, fg_color="transparent")
        val_frame.pack(fill="x", padx=15, pady=(0, 10))

        icon_lbl = ctk.CTkLabel(val_frame, text=icon, font=ctk.CTkFont(size=30))
        icon_lbl.pack(side="left", padx=(0, 10))

        value_lbl = ctk.CTkLabel(
            val_frame,
            text=value_text,
            font=ctk.CTkFont(size=20, weight="bold")
        )
        value_lbl.pack(side="left")

        return {"value": value_lbl, "icon": icon_lbl}

    def start_camera(self):
        """카메라 시작"""
        if self.is_running:
            return

        self.is_camera_loading = True
        self.show_loading("Camera Initializing...")
        self.start_btn.configure(state="disabled")
        threading.Thread(target=self._init_camera_thread, daemon=True).start()

    def _init_camera_thread(self):
        """카메라 초기화 스레드"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("No Webcam")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.is_running = True
            self.is_camera_loading = False

            self.after(0, lambda: self.status_label.configure(
                text="● System Running", text_color="#2CC985"
            ))
            self.after(0, self.hide_loading)
            self.after(0, self.update_video)

        except Exception as e:
            self.is_camera_loading = False
            self.after(0, lambda: self.status_label.configure(
                text="Error", text_color="red"
            ))
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, self.hide_loading)

    def stop_camera(self):
        """카메라 정지"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image=None)
        self.status_label.configure(text="System Stopped", text_color="gray")
        self.start_btn.configure(state="normal")
        self.update_dashboard(None)

    def update_dashboard(self, result):
        """우측 정보 패널 업데이트"""
        # 모든 얼굴 결과 가져오기
        all_results = self.analyzer.get_all_results()
        face_count = len(all_results)
        self.card_faces['value'].configure(text=f"{face_count} 명")
        
        # 감정 매핑
        emotion_map = {
            'angry': ('화남', '😡'),
            'disgust': ('혐오', '🤢'),
            'fear': ('두려움', '😨'),
            'happy': ('행복', '😄'),
            'sad': ('슬픔', '😢'),
            'surprise': ('놀람', '😲'),
            'neutral': ('평온', '😐')
        }

        # 기존 카드 중 제거할 것 찾기 (얼굴 수가 줄어든 경우)
        current_face_count = len(self.face_cards)
        if face_count < current_face_count:
            for idx in range(face_count, current_face_count):
                if idx in self.face_cards:
                    self.face_cards[idx]['card'].destroy()
                    del self.face_cards[idx]

        # 각 얼굴마다 카드 업데이트 또는 생성
        for idx, face_result in enumerate(all_results):
            age = face_result.get('age', 0)
            gender = face_result.get('dominant_gender', '?')
            emotion = face_result.get('dominant_emotion', '?')

            emo_text, emo_icon = emotion_map.get(emotion.lower(), (emotion, '🤔'))
            gender_text = '남성' if gender == 'Man' else '여성' if gender == 'Woman' else gender
            gender_icon = '👨' if gender == 'Man' else '👩' if gender == 'Woman' else '👤'

            # 기존 카드가 있으면 업데이트, 없으면 생성
            if idx in self.face_cards:
                # 카드 업데이트
                card_data = self.face_cards[idx]
                card_data['header'].configure(
                    text=f"Face {idx + 1}",
                    text_color="#2CC985" if idx == 0 else "#FF6B6B"
                )
                card_data['age'].configure(text=f"{age}세")
                card_data['gender_icon'].configure(text=gender_icon)
                card_data['gender'].configure(text=gender_text)
                card_data['emotion_icon'].configure(text=emo_icon)
                card_data['emotion'].configure(
                    text=emo_text,
                    text_color="#2CC985" if emotion == 'happy' else "white"
                )
            else:
                # 새 카드 생성
                face_card = self.create_face_card(
                    self.faces_scroll_frame,
                    idx + 1,
                    age,
                    gender_text,
                    gender_icon,
                    emo_text,
                    emo_icon,
                    emotion == 'happy'
                )
                self.face_cards[idx] = face_card

    def create_face_card(self, parent, face_num, age, gender_text, gender_icon, emotion_text, emotion_icon, is_happy):
        """개별 얼굴 정보 카드 생성"""
        card = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=8)
        card.pack(fill="x", padx=10, pady=5)

        # 얼굴 번호 헤더
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        header_label = ctk.CTkLabel(
            header_frame,
            text=f"Face {face_num}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2CC985" if face_num == 1 else "#FF6B6B"
        )
        header_label.pack(side="left")

        # 정보 프레임
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        # 나이
        age_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        age_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(age_frame, text="🎂", font=ctk.CTkFont(size=16)).pack(side="left", padx=(0, 5))
        age_label = ctk.CTkLabel(age_frame, text=f"{age}세", font=ctk.CTkFont(size=12))
        age_label.pack(side="left")

        # 성별
        gender_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        gender_frame.pack(fill="x", pady=2)
        gender_icon_label = ctk.CTkLabel(gender_frame, text=gender_icon, font=ctk.CTkFont(size=16))
        gender_icon_label.pack(side="left", padx=(0, 5))
        gender_label = ctk.CTkLabel(gender_frame, text=gender_text, font=ctk.CTkFont(size=12))
        gender_label.pack(side="left")

        # 감정
        emotion_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        emotion_frame.pack(fill="x", pady=2)
        emotion_icon_label = ctk.CTkLabel(emotion_frame, text=emotion_icon, font=ctk.CTkFont(size=16))
        emotion_icon_label.pack(side="left", padx=(0, 5))
        emotion_label = ctk.CTkLabel(
            emotion_frame,
            text=emotion_text,
            font=ctk.CTkFont(size=12),
            text_color="#2CC985" if is_happy else "white"
        )
        emotion_label.pack(side="left")

        return {
            "card": card,
            "header": header_label,
            "age": age_label,
            "gender_icon": gender_icon_label,
            "gender": gender_label,
            "emotion_icon": emotion_icon_label,
            "emotion": emotion_label
        }

    def update_video(self):
        """비디오 프레임 업데이트"""
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_video)
            return

        # 카메라 효과 적용 (좌우반전, 밝기, 대비, 흑백)
        frame = self._apply_camera_effects(frame)

        # 분석 및 데이터 갱신
        self.analyzer.process_frame(frame)
        result = self.analyzer.get_result()  # 첫 번째 얼굴 (대시보드용)
        all_results = self.analyzer.get_all_results()  # 모든 얼굴
        self.update_dashboard(result)

        # 여러 얼굴 박스 그리기
        if self.show_overlay_var.get() and all_results:
            for idx, face_result in enumerate(all_results):
                region = face_result.get('region', {})
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

                if w > 0:
                    # 얼굴 박스 (여러 명을 구분하기 위해 색상 변경)
                    color = (44, 201, 133) if idx == 0 else (201, 44, 133)  # 첫 번째는 녹색, 나머지는 빨간색
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 얼굴 번호 표시
                    cv2.putText(frame, f"Face {idx + 1}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 반투명 배경 (첫 번째 얼굴만)
                    if idx == 0:
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (44, 201, 133), -1)
                        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
        self.prev_time = curr_time
        self.fps_label.configure(text=f"FPS: {int(fps)}")

        # 이미지 변환 및 출력
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # 창 크기에 맞춰 리사이즈 (Aspect Ratio 유지)
        display_w = self.video_container.winfo_width()
        display_h = self.video_container.winfo_height()

        if display_w > 10 and display_h > 10:
            img_ratio = img.width / img.height
            screen_ratio = display_w / display_h

            if screen_ratio > img_ratio:
                new_h = display_h
                new_w = int(new_h * img_ratio)
            else:
                new_w = display_w
                new_h = int(new_w / img_ratio)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(10, self.update_video)

    def show_loading(self, text):
        """로딩 표시"""
        self.loading_label.configure(text=text)
        self.loading_label.place(relx=0.5, rely=0.5, anchor="center")

    def hide_loading(self):
        """로딩 숨김"""
        self.loading_label.place_forget()

    def update_loading_status(self, msg: Optional[str]):
        """로딩 상태 업데이트"""
        if msg:
            self.show_loading(msg)
        elif not self.is_camera_loading:
            self.hide_loading()

    def on_closing(self):
        """앱 종료 시 처리"""
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
