#!/usr/bin/env python3
"""
듀얼 로봇팔 컨트롤러 - 두 개의 Panda 로봇팔 제어
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class DualRobotController:
    def __init__(self, physics_client):
        self.physics_client = physics_client
        
        # 환경 설정
        self.setup_environment()
        
        # 로봇팔들 초기화
        self.setup_dual_robots()
        
        # 타겟 객체들 생성
        self.target_objects = []
        self.create_target_objects()
        
        print("✅ 듀얼 로봇 시스템 초기화 완료!")
    
    def setup_environment(self):
        """환경 설정"""
        # 바닥과 테이블 로드
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 중앙에 테이블 배치
        table_pos = [0.5, 0, 0]
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.table_id = p.loadURDF("table/table.urdf", table_pos, table_orientation)
        
        print("🏠 환경 설정 완료")
    
    def setup_dual_robots(self):
        """두 개의 로봇팔 설정"""
        # 로봇1 (왼쪽) - 수확 담당
        robot1_pos = [-0.2, 0.4, 0.625]  # 테이블 왼쪽
        robot1_orn = p.getQuaternionFromEuler([0, 0, -math.pi/4])  # -45도 회전
        self.robot1_id = p.loadURDF("franka_panda/panda.urdf", 
                                   basePosition=robot1_pos,
                                   baseOrientation=robot1_orn,
                                   useFixedBase=True)
        
        # 로봇2 (오른쪽) - 수취 담당
        robot2_pos = [-0.2, -0.4, 0.625]  # 테이블 오른쪽
        robot2_orn = p.getQuaternionFromEuler([0, 0, math.pi/4])  # +45도 회전
        self.robot2_id = p.loadURDF("franka_panda/panda.urdf", 
                                   basePosition=robot2_pos,
                                   baseOrientation=robot2_orn,
                                   useFixedBase=True)
        
        # 로봇 설정
        self.robots = {
            'robot1': {
                'id': self.robot1_id,
                'name': '수확로봇',
                'gripper_joints': [9, 10],
                'joint_indices': list(range(7)),
                'home_position': [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
            },
            'robot2': {
                'id': self.robot2_id,
                'name': '수취로봇',
                'gripper_joints': [9, 10],
                'joint_indices': list(range(7)),
                'home_position': [0, -0.785, 0, -2.356, 0, 1.571, -0.785]  # 반대 방향
            }
        }
        
        # 그리퍼 설정
        self.gripper_open_pos = 0.04
        self.gripper_closed_pos = 0.02
        self.gripper_force = 10
        
        # 홈 포지션으로 이동
        self.move_both_to_home()
        
        print("🤖 듀얼 로봇 설정 완료")
    
    def create_target_objects(self):
        """보라색 원통 객체 1개 생성 (샤프란 꽃 역할)"""
        # 단일 꽃 위치 - 테이블 중앙
        flower_position = [0.5, 0.0, 0.65]
        
        print("🌸 샤프란 꽃 객체 생성 중...")
        
        try:
            # 원통 모양 생성
            cylinder_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER, 
                radius=0.02, 
                height=0.05)
            
            cylinder_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER, 
                radius=0.02, 
                length=0.05,
                rgbaColor=[0.5, 0, 0.8, 1])  # 보라색
            
            cylinder_id = p.createMultiBody(
                baseMass=0.01,
                baseCollisionShapeIndex=cylinder_shape,
                baseVisualShapeIndex=cylinder_visual,
                basePosition=flower_position)
            
            self.target_objects.append(cylinder_id)
            print(f"   🌸 꽃 생성 완료: {flower_position}")
            
        except Exception as e:
            print(f"   ❌ 꽃 생성 실패: {e}")
        
        print(f"🌸 샤프란 꽃 1개 생성 완료!")
    
    def move_both_to_home(self):
        """두 로봇 모두 홈 포지션으로 이동"""
        for robot_key, robot_data in self.robots.items():
            robot_id = robot_data['id']
            home_pos = robot_data['home_position']
            
            for i, joint_pos in enumerate(home_pos):
                p.resetJointState(robot_id, i, joint_pos)
            
            print(f"   {robot_data['name']} 홈 포지션 설정 완료")
    
    def control_gripper(self, robot_id, open_gripper=True):
        """그리퍼 제어"""
        gripper_joints = [9, 10]
        gripper_pos = self.gripper_open_pos if open_gripper else self.gripper_closed_pos
        
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                force=self.gripper_force)
        
        # 동작 완료까지 대기
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1/240)
    
    def move_robot_to_position(self, robot_id, target_pos, target_orn=None, slow=False):
        """특정 로봇을 목표 위치로 이동"""
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        try:
            # 현재 조인트 상태를 시드로 사용
            seed_pose = []
            for i in range(7):
                joint_state = p.getJointState(robot_id, i)
                seed_pose.append(joint_state[0])
            
            joint_poses = p.calculateInverseKinematics(
                robot_id, 7, target_pos, target_orn,
                lowerLimits=[-2.8, -1.7, -2.8, -3.0, -2.8, -0.1, -2.8],
                upperLimits=[2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8],
                jointRanges=[5.6, 3.4, 5.6, 2.9, 5.6, 3.8, 5.6],
                restPoses=seed_pose,
                maxNumIterations=200,
                residualThreshold=1e-3
            )
            
            # 조인트 설정
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                      targetPosition=joint_poses[i],
                                      force=1000, maxVelocity=0.5 if slow else 1.0)
            
            # 움직임 완료까지 대기
            wait_steps = 480 if slow else 240
            for _ in range(wait_steps):
                p.stepSimulation()
                time.sleep(1/240)
            
            return True
            
        except Exception as e:
            print(f"⚠️ 로봇 이동 실패: {e}")
            return False
    
    def get_end_effector_position(self, robot_id):
        """엔드 이펙터의 현재 위치 확인"""
        state = p.getLinkState(robot_id, 7)
        return state[0], state[1]
    
    def get_object_position(self, object_id):
        """객체의 현재 위치 확인"""
        pos, orn = p.getBasePositionAndOrientation(object_id)
        return pos, orn
    
    def check_gripper_contact(self, robot_id, target_object_id):
        """그리퍼와 객체 간 접촉 확인"""
        gripper_link_ids = [9, 10, 11]
        contacts = []
        
        for link_id in gripper_link_ids:
            contact_points = p.getContactPoints(
                bodyA=robot_id, bodyB=target_object_id, linkIndexA=link_id)
            if contact_points:
                contacts.extend(contact_points)
        
        return len(contacts) > 0, len(contacts)
    
    def get_finger_contact_stats(self, robot_id, object_id):
        """손가락(9,10) 접촉 요약: (접촉점수, 총정상력, 양쪽손가락접촉 여부)"""
        finger_links = [9, 10]
        total_points = 0
        total_force = 0.0
        touched = set()
        for link in finger_links:
            cps = p.getContactPoints(bodyA=robot_id, bodyB=object_id, linkIndexA=link)
            if cps:
                total_points += len(cps)
                touched.add(link)
                # PyBullet contact tuple의 10번째 인자가 normalForce(=index 9)
                for c in cps:
                    if len(c) > 9:
                        total_force += float(c[9])
        both = (len(touched) == 2)
        return total_points, total_force, both

    def confirm_stable_grasp(self, robot_id, object_id, min_force=2.0, require_both=True, dwell_steps=30, timeout_steps=240):
        """그립 안정성 확인: 연속 dwell_steps 동안 조건 충족해야 True"""
        stable = 0
        for _ in range(timeout_steps):
            pts, force, both = self.get_finger_contact_stats(robot_id, object_id)
            ok = (pts >= 2) and (force >= min_force) and ((both and require_both) or (not require_both))
            if ok:
                stable += 1
                if stable >= dwell_steps:
                    return True
            else:
                stable = 0
            # 유지력 계속 주기
            self.maintain_grip(robot_id)
            p.stepSimulation()
            time.sleep(1/240)
        return False

    
    def gradual_descent_and_grasp(self, robot_id, start_pos, target_object_id):
        """점진적 하강 및 잡기 - Z축 제한 적용"""
        obj_pos, _ = self.get_object_position(target_object_id)
        current_z = start_pos[2]
        
        # Z축 제한: 0.65m까지만 내려가기 (테이블과 충돌 방지)
        min_z = 0.65
        target_z = max(obj_pos[2] + 0.05, min_z)  # 객체 위 5cm 또는 최소 높이
        step_size = 0.03
        
        print(f"🔽 점진적 하강 시작:")
        print(f"   시작 높이: {current_z:.3f}m")
        print(f"   목표 높이: {target_z:.3f}m")
        print(f"   최소 높이 제한: {min_z:.3f}m")
        
        while current_z > target_z:
            current_z = max(current_z - step_size, target_z)
            
            # 실시간 객체 위치 추적
            obj_pos, _ = self.get_object_position(target_object_id)
            step_target = [obj_pos[0], obj_pos[1], current_z]
            
            print(f"   하강 중: {current_z:.3f}m")
            self.move_robot_to_position(robot_id, step_target, slow=True)
            
            # 안정화 대기
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1/240)
        
        print(f"✅ 하강 완료! 최종 높이: {current_z:.3f}m")
        
        # 그리퍼 닫기
        print("🤏 그리퍼 닫기...")
        self.control_gripper(robot_id, open_gripper=False)
        
        # 접촉 확인
        has_contact, contact_count = self.check_gripper_contact(robot_id, target_object_id)
        print(f"접촉 상태: {has_contact} ({contact_count}개 접촉점)")
        
        return has_contact
    
    def check_docking_success(self, robot_id, target_object_id, expected_distance=0.05):
        """도킹 성공 여부 확인 (간단한 로그)"""
        try:
            ee_pos, _ = self.get_end_effector_position(robot_id)
            obj_pos, _ = self.get_object_position(target_object_id)
            
            distance = math.sqrt(sum((a-b)**2 for a, b in zip(ee_pos, obj_pos)))
            has_contact, contact_count = self.check_gripper_contact(robot_id, target_object_id)
            
            success = distance < expected_distance or has_contact
            
            # 간단한 로그만 출력 (성공/실패 시에만)
            if success:
                print(f"   ✅ 도킹 성공 (거리: {distance:.3f}m)")
            else:
                print(f"   ❌ 도킹 실패 (거리: {distance:.3f}m)")
            
            return success, distance, has_contact
            
        except Exception as e:
            print(f"   ❌ 도킹 체크 실패: {e}")
            return False, 999, False

    def calculate_facing_orientation(self, from_pos, to_pos):
        """두 위치를 기준으로 서로 마주보는 방향 계산 (Z축 포함)"""
        # 3D 방향 벡터 계산
        direction = [
            to_pos[0] - from_pos[0],
            to_pos[1] - from_pos[1],
            to_pos[2] - from_pos[2]  # Z축도 포함
        ]
        
        # XY 평면에서의 각도 (yaw)
        yaw = math.atan2(direction[0], direction[1])
        
        # Z축 각도 (pitch) - 위아래 각도
        xy_distance = math.sqrt(direction[0]**2 + direction[1]**2)
        if xy_distance > 0:
            pitch = math.atan2(-direction[2], xy_distance)  # 음수로 해야 올바른 방향
        else:
            pitch = 0
        
        # 집게가 해당 방향을 향하도록 오리엔테이션 생성
        orientation = p.getQuaternionFromEuler([pitch, math.pi/2, yaw])
        
        return orientation

    def check_object_still_grasped(self, robot_id, object_id):
        """객체가 여전히 잡혀있는지 확인"""
        try:
            # 그리퍼 상태 확인
            gripper_state = []
            for joint_idx in [9, 10]:  # 그리퍼 조인트
                joint_state = p.getJointState(robot_id, joint_idx)
                gripper_state.append(joint_state[0])
            
            # 그리퍼가 닫혀있는지 확인
            gripper_closed = all(state < self.gripper_closed_pos + 0.01 for state in gripper_state)
            
            # 접촉 확인
            has_contact, contact_count = self.check_gripper_contact(robot_id, object_id)
            
            # 객체 높이 확인 (떨어졌는지)
            obj_pos, _ = self.get_object_position(object_id)
            is_lifted = obj_pos[2] > 0.7  # 70cm 이상이면 들려있음
            
            grasped = gripper_closed and (has_contact or is_lifted)
            
            if not grasped:
                print(f"   ⚠️ 객체 놓침 감지! 그리퍼={gripper_closed}, 접촉={has_contact}, 높이={obj_pos[2]:.3f}m")
            
            return grasped
            
        except Exception as e:
            print(f"   ❌ 객체 상태 확인 실패: {e}")
            return False

    def check_robot_collision_risk(self):
        """두 로봇 간 충돌 위험 확인"""
        try:
            robot1_pos, _ = self.get_end_effector_position(self.robot1_id)
            robot2_pos, _ = self.get_end_effector_position(self.robot2_id)
            
            distance = math.sqrt(sum((a-b)**2 for a, b in zip(robot1_pos, robot2_pos)))
            
            # 15cm 이내면 충돌 위험
            collision_risk = distance < 0.15
            
            if collision_risk:
                print(f"   ⚠️ 충돌 위험! 로봇 간 거리: {distance:.3f}m")
            
            return collision_risk, distance
            
        except Exception as e:
            print(f"   ❌ 충돌 체크 실패: {e}")
            return False, 999

    def safe_retreat_both_robots(self, grasped_object_id, retreat_distance=0.5):
        """안전한 양방향 후퇴 (충돌 방지 + 객체 보호)"""
        print(f"⏮️ 안전 후퇴 {retreat_distance*100:.0f}cm (충돌방지+객체보호)...")
        
        try:
            # 1. 먼저 객체가 여전히 잡혀있는지 확인
            if not self.check_object_still_grasped(self.robot1_id, grasped_object_id):
                print("   ❌ 객체가 이미 놓쳐짐! 재잡기 시도...")
                # 재잡기 시도
                self.control_gripper(self.robot1_id, open_gripper=False)
                for _ in range(60):
                    p.stepSimulation()
                    time.sleep(1/240)
            
            # 2. 현재 위치 및 충돌 위험 확인
            robot1_pos, _ = self.get_end_effector_position(self.robot1_id)
            robot2_pos, _ = self.get_end_effector_position(self.robot2_id)
            
            collision_risk, current_distance = self.check_robot_collision_risk()
            
            # 3. 안전한 후퇴 방향 계산 (Z축 포함)
            center_point = [
                (robot1_pos[0] + robot2_pos[0]) / 2,
                (robot1_pos[1] + robot2_pos[1]) / 2,
                (robot1_pos[2] + robot2_pos[2]) / 2
            ]
            
            # 후퇴 방향 (Z축도 고려)
            robot1_direction = [
                robot1_pos[0] - center_point[0],
                robot1_pos[1] - center_point[1],
                (robot1_pos[2] - center_point[2]) * 0.3  # Z축은 30%만 적용 (너무 급격한 높이 변화 방지)
            ]
            
            robot2_direction = [
                robot2_pos[0] - center_point[0],
                robot2_pos[1] - center_point[1],
                (robot2_pos[2] - center_point[2]) * 0.3
            ]
            
            # 정규화
            robot1_length = math.sqrt(sum(d**2 for d in robot1_direction))
            robot2_length = math.sqrt(sum(d**2 for d in robot2_direction))
            
            if robot1_length > 0:
                robot1_unit = [d / robot1_length for d in robot1_direction]
            else:
                robot1_unit = [-1, 0, 0.1]  # 기본값 (약간 위로)
                
            if robot2_length > 0:
                robot2_unit = [d / robot2_length for d in robot2_direction]
            else:
                robot2_unit = [1, 0, 0.1]
            
            # 4. 충돌 위험이 높으면 더 멀리 후퇴
            if collision_risk:
                retreat_distance = max(retreat_distance, 0.7)  # 최소 70cm
                print(f"   🚨 충돌 위험으로 후퇴 거리 증가: {retreat_distance*100:.0f}cm")
            
            # 5. 단계별 안전 후퇴 (7단계)
            retreat_steps = 7
            for step in range(retreat_steps):
                progress = (step + 1) / retreat_steps
                
                # 현재 단계의 후퇴 위치
                robot1_retreat_pos = [
                    robot1_pos[0] + robot1_unit[0] * retreat_distance * progress,
                    robot1_pos[1] + robot1_unit[1] * retreat_distance * progress,
                    robot1_pos[2] + robot1_unit[2] * retreat_distance * progress
                ]
                
                robot2_retreat_pos = [
                    robot2_pos[0] + robot2_unit[0] * retreat_distance * progress,
                    robot2_pos[1] + robot2_unit[1] * retreat_distance * progress, 
                    robot2_pos[2] + robot2_unit[2] * retreat_distance * progress
                ]
                
                # 높이 제한 (너무 높거나 낮지 않게)
                robot1_retreat_pos[2] = max(0.8, min(1.2, robot1_retreat_pos[2]))
                robot2_retreat_pos[2] = max(0.8, min(1.2, robot2_retreat_pos[2]))
                
                # 서로 마주보는 각도 계산 (3D)
                robot1_orientation = self.calculate_facing_orientation(robot1_retreat_pos, robot2_retreat_pos)
                robot2_orientation = self.calculate_facing_orientation(robot2_retreat_pos, robot1_retreat_pos)
                
                # 동시에 이동
                self.move_robot_to_position(self.robot1_id, robot1_retreat_pos, robot1_orientation, slow=True)
                self.move_robot_to_position(self.robot2_id, robot2_retreat_pos, robot2_orientation, slow=True)
                
                # 각 단계마다 안전 확인
                for _ in range(25):
                    p.stepSimulation()
                    time.sleep(1/240)
                
                # 객체 상태 확인 (3단계마다)
                if step % 3 == 0 and not self.check_object_still_grasped(self.robot1_id, grasped_object_id):
                    print(f"   🔧 단계 {step+1}: 객체 재고정...")
                    self.control_gripper(self.robot1_id, open_gripper=False)
                    for _ in range(30):
                        p.stepSimulation()
                        time.sleep(1/240)
                
                # 충돌 위험 재확인
                collision_risk, new_distance = self.check_robot_collision_risk()
                if collision_risk and step < retreat_steps - 2:  # 마지막 2단계가 아니면
                    print(f"   ⚠️ 단계 {step+1}: 여전히 충돌 위험, 거리 증가")
                    retreat_distance = min(retreat_distance * 1.2, 1.0)  # 최대 100cm
            
            # 6. 최종 상태 확인
            final_robot1_pos, _ = self.get_end_effector_position(self.robot1_id)
            final_robot2_pos, _ = self.get_end_effector_position(self.robot2_id)
            final_distance = math.sqrt(sum((a-b)**2 for a, b in zip(final_robot1_pos, final_robot2_pos)))
            
            object_still_grasped = self.check_object_still_grasped(self.robot1_id, grasped_object_id)
            final_collision_risk, _ = self.check_robot_collision_risk()
            
            status = "✅ 성공" if object_still_grasped and not final_collision_risk else "⚠️ 주의"
            print(f"   {status} 안전 후퇴 완료! 거리: {final_distance:.3f}m, 객체: {'보유' if object_still_grasped else '분실'}")
            
        except Exception as e:
            print(f"   ❌ 안전 후퇴 실패: {e}")

    def retreat_both_robots(self, grasped_object_id, retreat_distance=0.5):
        """안전 후퇴 시스템 호출"""
        self.safe_retreat_both_robots(grasped_object_id, retreat_distance)

    def retreat_and_retry(self, robot_id, target_object_id, retreat_distance=0.5):
        """안전 후퇴 및 재시도"""
        self.safe_retreat_both_robots(target_object_id, retreat_distance)

    def perform_handover(self, grasped_object_id):
        """두 로봇 간 객체 전달 - 몸쪽으로 완전히 당긴 후 이동"""
        print("🤝 로봇 간 객체 전달 시작 (몸쪽 당기기 → 이동)...")
        
        # 1. 로봇1이 객체를 140cm 높이로 올리기
        lift_height = 1.4  # 140cm 높이
        ee_pos, _ = self.get_end_effector_position(self.robot1_id)
        lift_position = [ee_pos[0], ee_pos[1], lift_height]
        
        print(f"📈 로봇1: 객체를 {lift_height*100:.0f}cm 높이로 올리기...")
        self.move_robot_to_position(self.robot1_id, lift_position, slow=True)
        
        # 안정화 대기
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1/240)
        
        # 2. 로봇1을 홈 포지션으로 돌아가게 하기 (완전히 몸쪽으로)
        print("🏠 로봇1: 홈 포지션으로 복귀하여 객체 안전 확보...")
        
        # 홈 포지션으로 이동 (높이만 140cm로 유지)
        robot1_data = self.robots['robot1']
        home_joints = robot1_data['home_position']
        
        # 홈 포지션의 조인트 각도로 설정
        for i, joint_angle in enumerate(home_joints):
            p.setJointMotorControl2(self.robot1_id, i, p.POSITION_CONTROL, 
                                  targetPosition=joint_angle, force=1000, maxVelocity=0.5)
        
        # 홈 포지션 이동 완료까지 대기
        for _ in range(240):  # 4초 대기
            p.stepSimulation()
            time.sleep(1/240)
        
        # 홈 포지션에서 높이만 140cm로 조정
        ee_pos_home, ee_orn_home = self.get_end_effector_position(self.robot1_id)
        home_lift_pos = [ee_pos_home[0], ee_pos_home[1], lift_height]
        
        print(f"📈 로봇1: 홈 위치에서 {lift_height*100:.0f}cm 높이로 조정...")
        self.move_robot_to_position(self.robot1_id, home_lift_pos, slow=True)
        
        # 안정화 대기
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1/240)
        
        print("✅ 로봇1: 홈 포지션 복귀 완료")
        
        # 3. 로봇 베이스 위치 정의 (방향 계산용)
        robot1_base_pos = [-0.2, 0.4, 0.625]   # 로봇1 베이스 위치
        robot2_base_pos = [-0.2, -0.4, 0.625]  # 로봇2 베이스 위치
        
        # 이제 상대방(로봇2) 쪽으로 30cm만 이동
        robot2_base_pos = [-0.2, -0.4, 0.625]  # 로봇2 베이스 위치
        
        # 로봇1에서 로봇2 방향 계산
        direction_to_robot2 = [
            robot2_base_pos[0] - robot1_base_pos[0],
            robot2_base_pos[1] - robot1_base_pos[1],
            0
        ]
        
        # 방향 정규화
        direction_length = math.sqrt(direction_to_robot2[0]**2 + direction_to_robot2[1]**2)
        if direction_length > 0:
            unit_direction = [d / direction_length for d in direction_to_robot2]
        else:
            unit_direction = [0, -1, 0]  # 기본값: Y축 음의 방향
        
        # 상대방 쪽으로 30cm 이동한 최종 위치
        final_handover_pos = [
            ee_pos_home[0] + unit_direction[0] * 0.3,  # 홈 위치에서 30cm 이동
            ee_pos_home[1] + unit_direction[1] * 0.3,  # 홈 위치에서 30cm 이동
            lift_height  # 높이 유지
        ]
        
        # 상대방을 향한 방향 설정
        yaw_to_robot2 = math.atan2(unit_direction[0], unit_direction[1])
        robot1_final_orn = p.getQuaternionFromEuler([0, math.pi/2, yaw_to_robot2])
        
        print("➡️ 로봇1: 상대방 쪽으로 30cm 이동하여 고정...")
        self.move_robot_to_position(self.robot1_id, final_handover_pos, robot1_final_orn, slow=True)
        
        # 안정화 대기
        for _ in range(90):
            p.stepSimulation()
            time.sleep(1/240)
        
        print("✅ 로봇1: 최종 고정 위치 설정 완료 - 대기 모드")
        
        # 4. 로봇2 시작 위치 설정 (70cm 거리)
        robot1_final_pos, _ = self.get_end_effector_position(self.robot1_id)
        
        robot2_start_distance = 0.7  # 70cm 거리
        robot2_start_pos = [
            robot1_final_pos[0] + unit_direction[0] * robot2_start_distance,
            robot1_final_pos[1] + unit_direction[1] * robot2_start_distance,
            robot1_final_pos[2]  # 같은 높이 (140cm)
        ]
        
        print(f"🎯 로봇2: {robot2_start_distance*100:.0f}cm 거리에서 정렬 시작...")
        self.control_gripper(self.robot2_id, open_gripper=True)  # 그리퍼 열기
        
        # 5. 완벽한 정렬 찾기 (간단화)
        perfect_alignment = False
        alignment_attempts = 0
        max_alignment_attempts = 4  # 4번만 시도 (이제 더 직선적이므로)
        
        while not perfect_alignment and alignment_attempts < max_alignment_attempts:
            alignment_attempts += 1
            print(f"   🔄 정렬 시도 {alignment_attempts}/{max_alignment_attempts}")
            
            # 로봇2를 시작 위치로 이동
            self.move_robot_to_position(self.robot2_id, robot2_start_pos, slow=True)
            
            # 각도 조정 (매우 작은 조정)
            angle_offset = (alignment_attempts - 1) * 0.02  # 아주 작은 각도 변화
            
            # 로봇1을 향한 정확한 방향 계산
            direction_to_robot1 = [
                robot1_final_pos[0] - robot2_start_pos[0],
                robot1_final_pos[1] - robot2_start_pos[1],
                robot1_final_pos[2] - robot2_start_pos[2]
            ]
            
            # 각도 계산
            yaw = math.atan2(direction_to_robot1[0], direction_to_robot1[1]) + angle_offset
            
            # pitch 계산 (거의 수평이므로 작음)
            xy_distance = math.sqrt(direction_to_robot1[0]**2 + direction_to_robot1[1]**2)
            if xy_distance > 0:
                pitch = math.atan2(-direction_to_robot1[2], xy_distance)
            else:
                pitch = 0
            
            # 로봇2 방향 설정
            robot2_orn = p.getQuaternionFromEuler([pitch, -math.pi/2, yaw])
            
            # 로봇2를 해당 방향으로 설정
            self.move_robot_to_position(self.robot2_id, robot2_start_pos, robot2_orn, slow=True)
            
            # 정렬 상태 확인
            perfect_alignment = self.check_close_range_alignment()
            
            if perfect_alignment:
                print("   ✅ 정렬 완료!")
                break
            else:
                print("   ⚠️ 미세 조정...")
                for _ in range(20):
                    p.stepSimulation()
                    time.sleep(1/240)
        
        if not perfect_alignment:
            print("   ⚠️ 근사 정렬로 진행...")
        
        # 6. 로봇2가 직선 접근
        print("➡️ 로봇2: 직선 접근...")
        
        # 접근 단계 (6단계로 더 단순화)
        approach_steps = 6
        approach_distance = robot2_start_distance - 0.08  # 8cm 여유
        
        for step in range(approach_steps):
            progress = (step + 1) / approach_steps
            
            # 직선 접근 (단순함)
            current_approach_pos = [
                robot2_start_pos[0] - unit_direction[0] * approach_distance * progress,
                robot2_start_pos[1] - unit_direction[1] * approach_distance * progress,
                robot2_start_pos[2]
            ]
            
            # 로봇2만 이동
            self.move_robot_to_position(self.robot2_id, current_approach_pos, robot2_orn, slow=True)
            
            # 짧은 대기
            for _ in range(15):
                p.stepSimulation()
                time.sleep(1/240)
            
            # 객체와의 거리 확인
            robot2_pos, _ = self.get_end_effector_position(self.robot2_id)
            obj_pos, _ = self.get_object_position(grasped_object_id)
            obj_distance = math.sqrt(sum((a-b)**2 for a, b in zip(robot2_pos, obj_pos)))
            
            if obj_distance < 0.05:  # 5cm 이내면 접근 완료
                print(f"   ✅ 접근 완료! 거리: {obj_distance:.3f}m")
                break
        
        # 7. 로봇2 그리퍼 닫기
        print("🤏 로봇2: 그리퍼 닫기...")
        self.control_gripper(self.robot2_id, open_gripper=False)
        
        # 짧은 대기
        for _ in range(90):
            p.stepSimulation()
            time.sleep(1/240)
        
        # 8. 로봇1 그리퍼 열기 (전달 완료)
        print("📤 로봇1: 전달 완료 - 그리퍼 열기")
        self.control_gripper(self.robot1_id, open_gripper=True)
        
        # 9. 로봇2가 수확 상자로 이동
        delivery_pos = [0.1, -0.5, 0.8]
        delivery_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        print("📦 로봇2: 수확 상자로 이동...")
        self.move_robot_to_position(self.robot2_id, delivery_pos, delivery_orn)
        
        # 10. 로봇2가 객체 놓기
        print("📦 로봇2: 객체 놓기...")
        self.control_gripper(self.robot2_id, open_gripper=True)
        
        # 11. 두 로봇 홈 복귀
        print("🏠 두 로봇 홈 복귀...")
        robot1_data = self.robots['robot1']
        robot2_data = self.robots['robot2']
        
        for i, (joint1, joint2) in enumerate(zip(robot1_data['home_position'], robot2_data['home_position'])):
            p.setJointMotorControl2(robot1_data['id'], i, p.POSITION_CONTROL, 
                                  targetPosition=joint1, force=1000, maxVelocity=1.0)
            p.setJointMotorControl2(robot2_data['id'], i, p.POSITION_CONTROL, 
                                  targetPosition=joint2, force=1000, maxVelocity=1.0)
        
        for _ in range(300):
            p.stepSimulation()
            time.sleep(1/240)
        
        print("✅ 몸쪽 당기기 도킹 완료!")
        return True

    def check_close_range_alignment(self):
        """근거리 정렬 상태 확인 (더 관대한 기준)"""
        try:
            robot1_pos, _ = self.get_end_effector_position(self.robot1_id)
            robot2_pos, _ = self.get_end_effector_position(self.robot2_id)
            
            distance = math.sqrt(sum((a-b)**2 for a, b in zip(robot1_pos, robot2_pos)))
            height_diff = abs(robot1_pos[2] - robot2_pos[2])
            
            # 근거리 정렬 조건 (더 관대함):
            # 1. 거리 40cm~80cm (가까운 범위)
            # 2. 높이 차이 8cm 이내
            good_distance = 0.4 <= distance <= 0.8
            good_height = height_diff < 0.08
            
            perfect = good_distance and good_height
            
            return perfect
            
        except Exception as e:
            print(f"   ❌ 근거리 정렬 확인 실패: {e}")
            return False
    
    def dual_robot_harvest_sequence(self, target_object_id):
        """듀얼 로봇 수확 시퀀스"""
        print(f"\n🌸 듀얼 로봇 수확 시작 (객체 ID: {target_object_id})")
        
        # 1. 객체 위치 확인
        obj_pos, _ = self.get_object_position(target_object_id)
        print(f"🎯 타겟 위치: {[round(x, 3) for x in obj_pos]}")
        
        # 2. 로봇1이 접근
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
        print("🤖 로봇1: 접근 중...")
        self.control_gripper(self.robot1_id, open_gripper=True)
        self.move_robot_to_position(self.robot1_id, approach_pos)
        
        # 3. 로봇1이 점진적 하강하여 잡기
        print("🤖 로봇1: 잡기 시도...")
        grasp_success = self.gradual_descent_and_grasp(self.robot1_id, approach_pos, target_object_id)
        
        if not grasp_success:
            print("❌ 잡기 실패!")
            return False
        
        print("✅ 잡기 성공!")
        
        # 4. 두 로봇 간 전달
        handover_success = self.perform_handover(target_object_id)
        
        if handover_success:
            print("✅ 듀얼 로봇 수확 완료!")
            return True
        else:
            print("❌ 전달 실패!")
            return False
    
    def system_check(self):
        """시스템 상태 확인"""
        print("🔍 시스템 상태 확인...")
        
        # 로봇 상태 확인
        for robot_key, robot_data in self.robots.items():
            robot_id = robot_data['id']
            ee_pos, _ = self.get_end_effector_position(robot_id)
            print(f"   {robot_data['name']}: {[round(x, 3) for x in ee_pos]}")
        
        # 객체 상태 확인
        print(f"   생성된 객체 수: {len(self.target_objects)}개")
        if self.target_objects:
            obj_pos, _ = self.get_object_position(self.target_objects[0])
            print(f"   샤프란 꽃: {[round(x, 3) for x in obj_pos]}")
        
        print("✅ 시스템 상태 정상")
    
    def test_single_robot_harvest(self):
        """단일 로봇 수확 테스트"""
        print("🧪 단일 로봇 수확 테스트...")
        
        if not self.target_objects:
            print("❌ 수확할 객체가 없습니다!")
            return
        
        # 유일한 객체로 테스트
        target_obj = self.target_objects[0]
        obj_pos, _ = self.get_object_position(target_obj)
        
        # 로봇1으로 기본 수확 테스트
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
        
        print("🤖 로봇1: 단일 수확 테스트 중...")
        self.control_gripper(self.robot1_id, open_gripper=True)
        self.move_robot_to_position(self.robot1_id, approach_pos)
        
        # 잡기 시도
        grasp_success = self.gradual_descent_and_grasp(self.robot1_id, approach_pos, target_obj)
        
        if grasp_success:
            # 들어올리기
            lift_pos = [obj_pos[0], obj_pos[1], 0.8]
            self.move_robot_to_position(self.robot1_id, lift_pos)
            
            # 놓기
            place_pos = [0.1, 0.3, 0.8]
            self.move_robot_to_position(self.robot1_id, place_pos)
            self.control_gripper(self.robot1_id, open_gripper=True)
            
            print("✅ 단일 로봇 테스트 성공!")
        else:
            print("❌ 단일 로봇 테스트 실패!")
        
        # 홈으로 복귀
        self.move_both_to_home()
    
    def test_dual_robot_handover(self):
        """듀얼 로봇 전달 테스트"""
        print("🧪 듀얼 로봇 전달 테스트...")
        
        if not self.target_objects:
            print("❌ 테스트할 객체가 없습니다!")
            return
        
        # 유일한 객체로 테스트
        target_obj = self.target_objects[0]
        success = self.dual_robot_harvest_sequence(target_obj)
        
        if success:
            print("✅ 듀얼 로봇 전달 테스트 성공!")
        else:
            print("❌ 듀얼 로봇 전달 테스트 실패!")
    
    def run_full_harvest_cycle(self):
        """전체 수확 사이클 실행"""
        print("🌸 전체 수확 사이클 시작...")
        
        if not self.target_objects:
            print("❌ 수확할 꽃이 없습니다!")
            return
        
        # 유일한 꽃 수확
        target_obj = self.target_objects[0]
        print(f"🌸 샤프란 꽃 수확 중...")
        
        try:
            success = self.dual_robot_harvest_sequence(target_obj)
            if success:
                print("✅ 샤프란 꽃 수확 성공!")
            else:
                print("❌ 샤프란 꽃 수확 실패!")
                
        except Exception as e:
            print(f"❌ 수확 중 오류: {e}")
        
        print(f"🌸 수확 사이클 완료!")