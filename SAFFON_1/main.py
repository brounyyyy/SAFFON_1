#!/usr/bin/env python3
"""
샤프란 수확 듀얼 로봇팔 시스템 - 메인 실행 파일 (개선된 버전)
"""

import pybullet as p
import pybullet_data
import time
from robot_controller import ImprovedDualRobotController  # 클래스 이름 변경
from vision_system import VisionSystem

def main():
    """메인 실행 함수"""
    try:
        print("개선된 샤프란 수확 듀얼 로봇팔 시스템 시작")
        print("=" * 60)
        
        # PyBullet 초기화
        physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)  # 실시간 시뮬레이션 비활성화
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0.5]
        )
        
        # 로봇 컨트롤러 초기화 (개선된 버전)
        print("개선된 듀얼 로봇 시스템 초기화 중...")
        robot_controller = ImprovedDualRobotController(physics_client)
        
        # 비전 시스템 초기화
        print("비전 시스템 초기화 중...")
        vision_system = VisionSystem()
        
        print("시스템 초기화 완료!")
        print("-" * 60)
        
        # 시스템 상태 확인
        robot_controller.system_check()
        
        # 사용자 입력 대기
        while True:
            print("\n명령을 선택하세요:")
            print("1. 카메라 뷰 저장")
            print("2. 단일 로봇 테스트")
            print("3. 정밀 도킹 테스트")
            print("4. 듀얼 로봇 협업 수확")
            print("5. 전체 수확 사이클")
            print("6. 비전 기반 객체 탐지")
            print("7. 적응형 핸드오버 테스트")
            print("8. 시스템 종료")
            
            choice = input("선택 (1-8): ").strip()
            
            if choice == "1":
                print("카메라 뷰 저장 중...")
                vision_system.save_all_camera_views(robot_controller.target_objects)
                
            elif choice == "2":
                print("단일 로봇 테스트 시작...")
                robot_controller.test_single_robot_harvest()
                
            elif choice == "3":
                print("정밀 도킹 테스트 시작...")
                robot_controller.test_precision_docking()
                
            elif choice == "4":
                print("듀얼 로봇 협업 테스트 시작...")
                robot_controller.test_dual_robot_handover()
                
            elif choice == "5":
                print("전체 수확 사이클 시작...")
                robot_controller.run_full_harvest_cycle()
                
            elif choice == "6":
                print("비전 기반 객체 탐지 테스트...")
                # 비전 시스템으로 객체 탐지
                analysis_result = vision_system.analyze_scene(robot_controller.target_objects)
                if analysis_result:
                    print(f"탐지 결과:")
                    print(f"   최적 카메라: {analysis_result['best_camera']}")
                    print(f"   탐지된 객체: {analysis_result['detected_count']}개")
                    for i, pos_info in enumerate(analysis_result['world_positions']):
                        world_pos = pos_info['world_pos']
                        print(f"   객체 {i+1}: {[round(x, 3) for x in world_pos]}")
                else:
                    print("객체 탐지 실패")
                
            elif choice == "7":
                print("적응형 핸드오버 테스트 시작...")
                if robot_controller.target_objects:
                    robot_controller.adaptive_handover(robot_controller.target_objects[0])
                else:
                    print("테스트할 객체가 없습니다!")
                
            elif choice == "8":
                print("시스템 종료...")
                break
                
            else:
                print("잘못된 선택입니다.")
                
        print("샤프란 수확 시스템을 종료합니다.")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # PyBullet 연결 종료
        try:
            p.disconnect()
            print("PyBullet 연결 종료 완료")
        except:
            pass

if __name__ == "__main__":
    main()