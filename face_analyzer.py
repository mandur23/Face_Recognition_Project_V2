"""
얼굴 분석 로직 모듈
Face Analysis Logic Module
"""

import threading
import time
from typing import Optional, Dict, List
from deepface import DeepFace


class FaceAnalyzer:
    """얼굴 분석을 위한 클래스"""
    
    def __init__(self, analysis_interval: int = 15, loading_callback=None):
        """
        Args:
            analysis_interval: 얼굴 분석을 수행할 프레임 간격 (기본값: 15프레임)
            loading_callback: 로딩 상태를 업데이트할 콜백 함수
        """
        self.analysis_interval = analysis_interval
        self.frame_count = 0
        self.last_result: Optional[Dict] = None  # 첫 번째 얼굴 (대시보드용)
        self.last_results: List[Dict] = []  # 모든 얼굴 결과
        self.result_timestamp: float = 0  # 결과가 생성된 시간
        self.result_ttl: float = 2.0  # 결과 유지 시간 (초)
        self.is_analyzing = False
        self.is_loading_model = False
        self.model_loaded = False  # 모델이 이미 로드되었는지 확인
        self.lock = threading.Lock()
        self.loading_callback = loading_callback  # 로딩 상태 콜백

    def _run_deepface(self, img):
        """DeepFace를 사용하여 얼굴 분석 수행"""
        try:
            # 첫 번째 분석일 때만 모델 로딩 표시
            if not self.model_loaded:
                if self.loading_callback:
                    self.loading_callback("모델 로딩 중...")
                self.is_loading_model = True
            
            # RetinaFace 백엔드 사용 (더 정확한 얼굴 감지)
            # RetinaFace가 없으면 opencv 사용
            try:
                objs = DeepFace.analyze(
                    img_path=img, 
                    actions=['age', 'gender', 'emotion'],
                    detector_backend='retinaface',  # 더 정확한 얼굴 감지
                    enforce_detection=False,
                    silent=True
                )
            except:
                # RetinaFace 실패 시 opencv 사용
                objs = DeepFace.analyze(
                    img_path=img, 
                    actions=['age', 'gender', 'emotion'],
                    detector_backend='opencv', 
                    enforce_detection=False,
                    silent=True
                )
            
            # 모델 로딩 완료 표시
            if not self.model_loaded:
                self.model_loaded = True
                self.is_loading_model = False
                if self.loading_callback:
                    self.loading_callback(None)  # 로딩 완료
            
            if objs:
                with self.lock:
                    # 모든 얼굴 결과 저장
                    self.last_results = objs if isinstance(objs, list) else [objs]
                    # 첫 번째 얼굴은 대시보드용으로 저장
                    self.last_result = self.last_results[0] if self.last_results else None
                    self.result_timestamp = time.time()  # 결과 생성 시간 기록
        except Exception as e:
            # 얼굴이 감지되지 않아도 에러로 처리하지 않음
            # 모델 로딩 중이었다면 완료 처리
            if not self.model_loaded:
                self.model_loaded = True
                self.is_loading_model = False
                if self.loading_callback:
                    self.loading_callback(None)
        finally:
            self.is_analyzing = False

    def process_frame(self, img):
        """프레임 처리 및 분석 요청"""
        self.frame_count += 1
        if not self.is_analyzing and self.frame_count % self.analysis_interval == 0:
            self.is_analyzing = True
            thread = threading.Thread(target=self._run_deepface, args=(img.copy(),))
            thread.daemon = True
            thread.start()

    def get_result(self):
        """마지막 분석 결과 반환 (TTL 체크 포함) - 첫 번째 얼굴"""
        with self.lock:
            # 결과가 있고, 아직 유효한 시간 내에 있으면 반환
            if self.last_result and (time.time() - self.result_timestamp) < self.result_ttl:
                return self.last_result
            # 결과가 오래되었으면 None 반환 (화면에서 사라짐)
            return None
    
    def get_all_results(self):
        """모든 얼굴 분석 결과 반환 (TTL 체크 포함)"""
        with self.lock:
            # 결과가 있고, 아직 유효한 시간 내에 있으면 반환
            if self.last_results and (time.time() - self.result_timestamp) < self.result_ttl:
                return self.last_results
            # 결과가 오래되었으면 빈 리스트 반환
            return []

