"""
비전 시스템 - 다중 카메라 및 객체 탐지
"""

import pybullet as p
import cv2
import numpy as np
import os

class VisionSystem:
    def __init__(self):
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 60
        self.camera_aspect = self.camera_width / self.camera_height
        self.camera_near = 0.1
        self.camera_far = 3.0
        
        # 다중 카메라 설정
        self.setup_cameras()
        print("비전 시스템 초기화 완료")
    
    def setup_cameras(self):
        """다중 카메라 위치 설정"""
        self.cameras = {
            'top_view': {
                'eye_pos': [0.5, 0, 1.8],      # 위에서 내려다보는 메인 카메라
                'target_pos': [0.5, 0, 0.65],
                'up_vector': [0, 1, 0],
                'description': '상단 뷰 (메인 카메라)'
            },
            'side_left': {
                'eye_pos': [0.2, 0.6, 1.2],   # 왼쪽에서 보는 카메라
                'target_pos': [0.5, 0, 0.65],
                'up_vector': [0, 0, 1],
                'description': '좌측 뷰'
            },
            'side_right': {
                'eye_pos': [0.2, -0.6, 1.2],  # 오른쪽에서 보는 카메라
                'target_pos': [0.5, 0, 0.65],
                'up_vector': [0, 0, 1],
                'description': '우측 뷰'
            },
            'front_view': {
                'eye_pos': [0.9, 0, 1.0],     # 앞쪽에서 보는 카메라
                'target_pos': [0.5, 0, 0.65],
                'up_vector': [0, 0, 1],
                'description': '전면 뷰'
            },
            'robot1_view': {
                'eye_pos': [-0.1, 0.5, 1.0],  # 로봇1 관점
                'target_pos': [0.5, 0, 0.65],
                'up_vector': [0, 0, 1],
                'description': '로봇1 관점'
            },
            'robot2_view': {
                'eye_pos': [-0.1, -0.5, 1.0], # 로봇2 관점
                'target_pos': [0.5, 0, 0.65],
                'up_vector': [0, 0, 1],
                'description': '로봇2 관점'
            }
        }
        
        print(f"{len(self.cameras)}개 카메라 설정 완료")
    
    def get_camera_image(self, camera_name='top_view'):
        """특정 카메라에서 이미지 획득"""
        if camera_name not in self.cameras:
            print(f"존재하지 않는 카메라: {camera_name}")
            return None, None, None
            
        camera = self.cameras[camera_name]
        
        try:
            view_matrix = p.computeViewMatrix(
                camera['eye_pos'], 
                camera['target_pos'], 
                camera['up_vector']
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                self.camera_fov,
                self.camera_aspect,
                self.camera_near,
                self.camera_far
            )
            
            camera_data = p.getCameraImage(
                self.camera_width, 
                self.camera_height, 
                viewMatrix=view_matrix, 
                projectionMatrix=projection_matrix
            )
            
            width, height, rgbPixels, depthPixels, segmentationMaskBuffer = camera_data
            
            # RGB 이미지를 OpenCV 형식으로 변환
            rgb_array = np.array(rgbPixels, dtype=np.uint8).reshape(height, width, 4)
            rgb_array = rgb_array[:, :, :3]  # RGBA에서 RGB로
            
            return rgb_array, depthPixels, segmentationMaskBuffer
            
        except Exception as e:
            print(f"카메라 {camera_name} 이미지 획득 오류: {e}")
            return None, None, None
    
    def get_all_camera_images(self):
        """모든 카메라에서 이미지 획득"""
        all_images = {}
        for camera_name in self.cameras.keys():
            rgb_img, depth_img, seg_img = self.get_camera_image(camera_name)
            if rgb_img is not None:
                all_images[camera_name] = {
                    'rgb': rgb_img,
                    'depth': depth_img,
                    'segmentation': seg_img
                }
        return all_images
    
    def detect_purple_objects(self, rgb_image):
        """보라색 객체 탐지 (색상 기반)"""
        try:
            # 이미지 데이터 타입 확인 및 변환
            if rgb_image.dtype != np.uint8:
                rgb_image = rgb_image.astype(np.uint8)
            
            # BGR로 변환 (OpenCV 사용)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            
            # 보라색 범위 정의 (더 넓은 범위)
            lower_purple1 = np.array([120, 50, 50])
            upper_purple1 = np.array([150, 255, 255])
            
            # 보라색 범위 2 (자주색 계열)
            lower_purple2 = np.array([140, 30, 30])
            upper_purple2 = np.array([170, 255, 255])
            
            # 두 범위의 마스크 생성
            mask1 = cv2.inRange(hsv_image, lower_purple1, upper_purple1)
            mask2 = cv2.inRange(hsv_image, lower_purple2, upper_purple2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # 최소 면적 필터 (더 작은 값)
                    # 객체 중심점 계산
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detected_objects.append({
                            'center': (cx, cy),
                            'area': area,
                            'contour': contour
                        })
                        
            return detected_objects, mask
            
        except Exception as e:
            print(f"색상 탐지 오류: {e}")
            return [], None
    
    def detect_objects_multi_camera(self, target_objects=None):
        """다중 카메라를 사용한 객체 탐지"""
        all_detections = {}
        
        for camera_name in self.cameras.keys():
            rgb_img, depth_img, seg_img = self.get_camera_image(camera_name)
            
            if rgb_img is not None:
                detected_objects, mask = self.detect_purple_objects(rgb_img)
                
                if detected_objects:
                    print(f"📷 {camera_name}: {len(detected_objects)}개 객체 탐지")
                    all_detections[camera_name] = {
                        'objects': detected_objects,
                        'mask': mask,
                        'depth': depth_img,
                        'rgb': rgb_img
                    }
        
        return all_detections
    
    def find_best_detection(self, all_detections):
        """가장 좋은 탐지 결과 선택"""
        best_camera = None
        best_detection = None
        max_score = 0
        
        for camera_name, detection in all_detections.items():
            # 점수 계산: 객체 수 + 면적 기준
            objects = detection['objects']
            if objects:
                total_area = sum(obj['area'] for obj in objects)
                score = len(objects) * 100 + total_area  # 객체 수를 더 중요하게
                
                if score > max_score:
                    max_score = score
                    best_camera = camera_name
                    best_detection = detection
        
        if best_detection:
            print(f"최적 카메라: {best_camera} (점수: {max_score})")
            return best_camera, best_detection
        
        return None, None
    
    def pixel_to_world_coordinates(self, pixel_x, pixel_y, camera_name='top_view', depth_value=None):
        """픽셀 좌표를 월드 좌표로 변환"""
        try:
            # 카메라별 변환 파라미터 (간단한 근사)
            if camera_name == 'top_view':
                # 위에서 내려다보는 카메라의 변환
                norm_x = (pixel_x - self.camera_width/2) / (self.camera_width/2)
                norm_y = (pixel_y - self.camera_height/2) / (self.camera_height/2)
                
                # 테이블 위의 대략적인 월드 좌표로 변환
                world_x = 0.5 + norm_x * 0.3  # 테이블 범위에 맞게 스케일링
                world_y = -norm_y * 0.3       # Y축 반전
                world_z = 0.70  # 테이블 높이 + 객체 높이
                
            else:
                # 다른 카메라들에 대한 근사 변환
                norm_x = (pixel_x - self.camera_width/2) / (self.camera_width/2)
                norm_y = (pixel_y - self.camera_height/2) / (self.camera_height/2)
                
                world_x = 0.5 + norm_x * 0.25
                world_y = norm_y * 0.25
                world_z = 0.70
            
            return [world_x, world_y, world_z]
            
        except Exception as e:
            print(f"좌표 변환 오류: {e}")
            return [0.5, 0, 0.7]  # 기본값 반환
    
    def save_camera_image(self, camera_name, rgb_image, detected_objects=None, save_dir="camera_images"):
        """카메라 이미지 저장"""
        try:
            # 저장 디렉토리 생성
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 원본 이미지 저장
            filename = os.path.join(save_dir, f"{camera_name}_original.png")
            cv2.imwrite(filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            
            # 탐지 결과가 있으면 표시하여 저장
            if detected_objects:
                result_img = rgb_image.copy()
                
                for i, obj in enumerate(detected_objects):
                    center = obj['center']
                    area = obj['area']
                    
                    # 중심점에 원 그리기
                    cv2.circle(result_img, center, 8, (0, 255, 0), 2)
                    
                    # 텍스트 추가
                    text = f"{i+1}({area})"
                    cv2.putText(result_img, text, 
                              (center[0] + 10, center[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 컨투어 그리기
                    cv2.drawContours(result_img, [obj['contour']], -1, (255, 0, 0), 2)
                
                detection_filename = os.path.join(save_dir, f"{camera_name}_detection.png")
                cv2.imwrite(detection_filename, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                
                return filename, detection_filename
            
            return filename, None
            
        except Exception as e:
            print(f"{camera_name} 이미지 저장 실패: {e}")
            return None, None
    
    def save_all_camera_views(self, target_objects=None):
        """모든 카메라 뷰 저장"""
        print("모든 카메라 뷰 저장 중...")
        
        save_dir = f"camera_images_{int(time.time())}"  # 타임스탬프 포함
        saved_files = []
        
        # 모든 카메라 이미지 획득
        all_images = self.get_all_camera_images()
        
        for camera_name, images in all_images.items():
            try:
                # 객체 탐지 수행
                detected_objects, mask = self.detect_purple_objects(images['rgb'])
                
                # 이미지 저장
                original_file, detection_file = self.save_camera_image(
                    camera_name, images['rgb'], detected_objects, save_dir)
                
                if original_file:
                    saved_files.append(original_file)
                    print(f"   {camera_name}: {original_file}")
                    
                    if detection_file:
                        saved_files.append(detection_file)
                        print(f"      탐지 결과: {len(detected_objects)}개 객체")
                
                # 마스크 이미지도 저장
                if mask is not None:
                    mask_filename = os.path.join(save_dir, f"{camera_name}_mask.png")
                    cv2.imwrite(mask_filename, mask)
                    saved_files.append(mask_filename)
                    
            except Exception as e:
                print(f"   {camera_name} 처리 실패: {e}")
                continue
        
        print(f"총 {len(saved_files)}개 파일 저장 완료")
        print(f"저장 위치: {save_dir}/")
        
        return saved_files
    
    def analyze_scene(self, target_objects=None):
        """전체 장면 분석"""
        print("장면 분석 중...")
        
        # 다중 카메라 탐지
        all_detections = self.detect_objects_multi_camera(target_objects)
        
        if not all_detections:
            print("객체를 탐지하지 못했습니다.")
            return None
        
        # 최적 탐지 결과 선택
        best_camera, best_detection = self.find_best_detection(all_detections)
        
        if not best_detection:
            print("유효한 탐지 결과가 없습니다.")
            return None
        
        # 월드 좌표로 변환
        world_positions = []
        for obj in best_detection['objects']:
            pixel_x, pixel_y = obj['center']
            world_pos = self.pixel_to_world_coordinates(pixel_x, pixel_y, best_camera)
            world_positions.append({
                'world_pos': world_pos,
                'pixel_pos': (pixel_x, pixel_y),
                'area': obj['area'],
                'camera': best_camera
            })
        
        analysis_result = {
            'best_camera': best_camera,
            'detected_count': len(world_positions),
            'world_positions': world_positions,
            'raw_detection': best_detection
        }
        
        print(f"장면 분석 완료:")
        print(f"   최적 카메라: {best_camera}")
        print(f"   탐지된 객체: {len(world_positions)}개")
        
        return analysis_result
    
    def get_camera_info(self):
        """카메라 정보 출력"""
        print("카메라 시스템 정보:")
        print(f"   해상도: {self.camera_width}×{self.camera_height}")
        print(f"   FOV: {self.camera_fov}°")
        print(f"   카메라 수: {len(self.cameras)}개")
        
        for name, camera in self.cameras.items():
            print(f"   {name}: {camera['description']}")
            print(f"      위치: {camera['eye_pos']}")
            print(f"      타겟: {camera['target_pos']}")


import time
