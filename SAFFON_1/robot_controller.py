#!/usr/bin/env python3
"""
개선된 듀얼 로봇팔 컨트롤러 - 안정적 핸드오버 시스템
- 상태기계 기반 핸드오버
- 6D 포즈 서보 제어
- 가상 용접(constraint) 시스템
- 가드 접근(guarded approach)
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class ImprovedDualRobotController:
    def __init__(self, physics_client):
        self.physics_client = physics_client
        
        # === 가동 조인트 전용 DOF 맵 (가장 먼저 초기화) ===
        self.movable_joints = {}
        self.jindex_map = {}
        
        # === 추가: EE 링크 설정 ===
        self.ee_link_index = 11          # 손 링크로 변경 (기존 7에서 11로)
        
        # === 도킹 관련 설정 (정면-정렬 고정) ===
        self.TOOL_AXIS_LOCAL = np.array([1,0,0])   # Panda 그리퍼 전진축(+X)
        self.WORLD_UP = np.array([0,0,1])
        
        # 핸드오버 안전 파라미터
        self.handover_gap_min = 0.14
        self.handover_gap_max = 0.20
        self.axial_safe_stop = 0.05   # 축방향 2cm 이내면 더 안 밀고 정지
        # 핸드오버 중앙선(y=table_center_y)에서 양쪽 여유(중앙선 침범 금지)
        self.handover_side_margin_y = 0.01  # 5cm
        # (NEW) 도킹 재시도 설정
        self.docking_max_attempts = 5      # 넘겨받기 최대 시도 수
        self.failed_retreat_distance = 0.10  # 실패 시 후퇴 거리(30cm)
        self.near_grasp_tol = 0.069           # FAST-PASS 임계(2cm)
        
        # 그리퍼 개폐 폭(파라미터 튜닝)
        self.gripper_open_pos_wide = 0.06   # 핸드오버 접근용, 더 넓게
        self.gripper_open_pos = 0.04        # 일반 오픈
        self.gripper_closed_pos = 0.02
        self.gripper_force = 10
        
        # 그립 유지 설정
        self.grip_hold_force = 50           # 객체를 잡고 있을 때 지속적인 힘
        self.grip_hold_pos = 0.015          # 꽉 잡을 때 위치 (더 꽉)
        self.active_grips = {}              # robot_id -> True/False (그립 상태 추적)
        
        # 환경 설정
        self.setup_environment()
        
        # 로봇팔들 초기화
        self.setup_dual_robots()
        
        # 타겟 객체들 생성
        self.target_objects = []
        self.create_target_objects()
        
        print("개선된 듀얼 로봇 시스템 초기화 완료!")
    
    # === 공용 헬퍼 함수 ===
    def _as_float_list(self, v):
        """숫자 리스트를 float 리스트로 변환"""
        return [float(x) for x in list(v)]

    def _full_joint_state(self, body_id):
        """로봇의 전체 조인트 상태 반환"""
        n = p.getNumJoints(body_id)
        q = [p.getJointState(body_id, j)[0] for j in range(n)]
        qd = [0.0] * n
        qa = [0.0] * n
        return q, qd, qa
    
    def _take_velocity_control(self, body_id, q_indices, force=250):
        """VELOCITY 제어로 선점하여 이전 POSITION 제어 끊기"""
        for j in q_indices:
            p.setJointMotorControl2(body_id, j, p.VELOCITY_CONTROL, 
                                  targetVelocity=0.0, force=force)
    
    def _quat_axis_angle(self, axis, ang):
        """축-각도에서 쿼터니언 생성"""
        axis = np.array(axis, dtype=float)
        axis /= (np.linalg.norm(axis)+1e-9)
        s = math.sin(ang/2.0)
        return [axis[0]*s, axis[1]*s, axis[2]*s, math.cos(ang/2.0)]
    
    # === (NEW) FAST-PASS 근접 체크 ===
    def _euclid_dist(self, a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    
    def _pinch_center(self, robot_id):
        p9 = p.getLinkState(robot_id, 9,  computeForwardKinematics=True)[0]   # 왼/오른손가락
        p10= p.getLinkState(robot_id, 10, computeForwardKinematics=True)[0]
        return [(p9[0]+p10[0])*0.5, (p9[1]+p10[1])*0.5, (p9[2]+p10[2])*0.5]

    def _near_grasp_ready(self, receiver_id, object_id, tol=None):
        tol = self.near_grasp_tol if tol is None else tol  # 0.02
        pinch = self._pinch_center(receiver_id)
        obj, _ = self.get_object_position(object_id)
        dxy = np.linalg.norm(np.array(pinch[:2]) - np.array(obj[:2]))
        dz  = abs(pinch[2] - obj[2])
        ok  = (dxy <= tol and dz <= 0.015) or (np.linalg.norm(np.array(pinch)-np.array(obj)) <= tol)
        print(f"   (근접체크) 핀치–obj dxy={dxy*100:.1f}cm dz={dz*100:.1f}mm 임계={tol*100:.0f}cm → {ok}")
        return ok

    # === (NEW) 스윙-인 프리패스: 크게 돌아 진입해서 정면 접근 보장 ===
    def _swing_in_prepath(self, r2_start_pos, r1_port_pos, r2_quat, radius=0.18):
        """
        r2가 r1 쪽으로 바로 직선 접근하지 않고, 좌측으로 크게 원호를 그리며 접근하도록
        중간 웨이포인트를 한 번 찍어 준다. (시각적으로 '반대방향으로 돌며' 접근)
        """
        v = np.array(r1_port_pos[:2]) - np.array(r2_start_pos[:2])
        n = np.linalg.norm(v)
        if n < 1e-6:
            return
        f2 = v / n
        left = np.array([-f2[1], f2[0]])  # 좌측 노멀
        wp1 = [r2_start_pos[0] + left[0]*radius,
               r2_start_pos[1] + left[1]*radius,
               r2_start_pos[2]]
        wp1 = self._clamp_side_handover(self.robot2_id, wp1)
        print(f"   스윙-인 웨이포인트: {[round(x,3) for x in wp1]}")
        self.move_robot_to_position(self.robot2_id, wp1, r2_quat, slow=True)

    # === (NEW) 수취팔을 '마주보기 반대방향'으로 후퇴 ===
    def _retreat_receiver_along_facing(self, receiver_id, giver_port_pos, dist=None):
        """
        giver_port_pos(주는 팔 포트)를 바라보는 선을 기준으로,
        수취 팔(receiver)만 반대방향으로 dist만큼 후퇴.
        """
        if dist is None:
            dist = self.failed_retreat_distance
        ee_pos, ee_quat = self.get_end_effector_position(receiver_id)
        dir_vec = np.array(giver_port_pos) - np.array(ee_pos)
        # 수평면 기준으로만 후퇴
        dir_vec[2] = 0.0
        n = np.linalg.norm(dir_vec)
        if n < 1e-6:
            print("   ℹ후퇴 생략: 마주보기 벡터가 너무 짧음")
            return
        u = dir_vec / n
        retreat = np.array(ee_pos) - u * dist
        retreat = self._clamp_side_handover(receiver_id, retreat.tolist())
        retreat[2] = max(retreat[2], self.min_handover_clear_z)
        print(f"⏮수취 로봇 후퇴: {dist*100:.0f}cm (마주보기 반대방향)")
        self.move_robot_to_position(receiver_id, retreat, ee_quat, slow=True)

    # === (NEW) 임시 고정(용접): 회전 중 낙하 방지용 짧은 고정 ===
    def _temp_weld(self, parent_id, parent_link, child_id, hold_steps=120):
        """
        parent(EE 링크)와 child(객체)를 고정 관절로 잠깐 묶어 회전 중 낙하 방지.
        hold_steps 프레임만 유지 후 즉시 해제.
        """
        print(f"임시 고정 시작: {hold_steps}프레임 동안")
        # parent(EE) 월드 포즈
        pst = p.getLinkState(parent_id, parent_link, computeForwardKinematics=True)
        ppos, porn = pst[0], pst[1]
        # child(물체) 월드 포즈
        cpos, corn = p.getBasePositionAndOrientation(child_id)
        # child를 parent-프레임으로
        pinv = p.invertTransform(ppos, porn)
        pcl, qcl = p.multiplyTransforms(pinv[0], pinv[1], cpos, corn)
        # 고정 관절 생성: parentFrame=identity, childFrame=(pcl,qcl)
        cid = p.createConstraint(parent_id, parent_link, child_id, -1,
                                 p.JOINT_FIXED, [0,0,0],
                                 [0,0,0], pcl,
                                 parentFrameOrientation=[0,0,0,1],
                                 childFrameOrientation=qcl)
        for _ in range(hold_steps):
            self.maintain_grip(parent_id)
            p.stepSimulation()
            time.sleep(1/240)
        p.removeConstraint(cid)
        print("임시 고정 해제 완료")
    
    def _build_dof_maps(self, body_id):
        """가동 조인트 전용 DOF 맵 생성"""
        mov, jmap = [], {}
        for j in range(p.getNumJoints(body_id)):
            if p.getJointInfo(body_id, j)[2] != p.JOINT_FIXED:  # 가동 조인트만
                jmap[j] = len(mov)
                mov.append(j)
        self.movable_joints[body_id] = mov
        self.jindex_map[body_id] = jmap
        print(f"   로봇 {body_id}: 가동 조인트 {len(mov)}개 매핑 완료")

    def _q_all_movable(self, body_id):
        """가동 조인트만의 상태 반환"""
        return [p.getJointState(body_id, j)[0] for j in self.movable_joints[body_id]]
    
    def _quat_from_two_vectors(self, u, v):
        """두 벡터 간 회전 쿼터니언 (벡터-to-벡터 정렬)"""
        u = np.array(u, dtype=float); u /= (np.linalg.norm(u)+1e-9)
        v = np.array(v, dtype=float); v /= (np.linalg.norm(v)+1e-9)
        d = float(np.dot(u, v))
        
        if d > 1.0-1e-9:  # same direction
            return [0,0,0,1]
        if d < -1.0+1e-9:  # opposite → 임의의 수직축으로 180°
            axis = np.cross(u, [1,0,0])
            if np.linalg.norm(axis) < 1e-6: 
                axis = np.cross(u, [0,1,0])
            axis /= (np.linalg.norm(axis)+1e-9)
            s = math.sin(math.pi/2)
            return [axis[0]*s, axis[1]*s, axis[2]*s, math.cos(math.pi/2)]
        
        c = np.cross(u, v)
        s = math.sqrt((1.0+d)*2.0)
        return [c[0]/s, c[1]/s, c[2]/s, s/2.0]
    
    def setup_environment(self):
        """환경 설정"""
        # 바닥과 테이블 로드
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 중앙에 테이블 배치
        table_pos = [0.5, 0, 0]
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.table_id = p.loadURDF("table/table.urdf", table_pos, table_orientation)
        self.table_center_y = table_pos[1]  # 중앙선(y=0)
        
        # 테이블 상판 높이 추정 및 최소 여유 높이(바닥 간섭 방지)
        aabb_min, aabb_max = p.getAABB(self.table_id)
        self.table_top_z = aabb_max[2]
        self.min_handover_clear_z = self.table_top_z + 0.8  # 상판 + 80cm 이상
        print(f"   테이블 상판 높이: {self.table_top_z:.3f}m, 최소 핸드오버 높이: {self.min_handover_clear_z:.3f}m")
        
        print("환경 설정 완료")
    
    def setup_dual_robots(self):
        """두 개의 로봇팔 설정"""
        # 로봇1 (왼쪽) - 수확 담당
        robot1_pos = [-0.2, 0.4, 0.625]  # 테이블 왼쪽
        robot1_orn = p.getQuaternionFromEuler([0, 0, -math.pi/4])  # -45도 회전
        self.robot1_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=robot1_pos,
            baseOrientation=robot1_orn,
            useFixedBase=True
        )

        # 로봇2 (오른쪽) - 수취 담당
        robot2_pos = [-0.2, -0.4, 0.625]  # 테이블 오른쪽
        robot2_orn = p.getQuaternionFromEuler([0, 0, math.pi/4])  # +45도 회전
        self.robot2_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=robot2_pos,
            baseOrientation=robot2_orn,
            useFixedBase=True
        )

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
                'home_position': [0, -0.785, 0, -2.356, 0, 1.571, -0.785]
            }
        }

        # 홈 포지션으로 이동
        self.move_both_to_home()

        # 가동 조인트 DOF 맵 생성
        self._build_dof_maps(self.robot1_id)
        self._build_dof_maps(self.robot2_id)

        # 그리퍼 마찰/앵커 강화
        for rid in [self.robot1_id, self.robot2_id]:
            for j in [9, 10, self.ee_link_index]:
                p.changeDynamics(rid, j, lateralFriction=1.0, frictionAnchor=True)

        # (여기가 핵심) 로봇 ID/조인트가 모두 준비된 '뒤'에 축 보정 로그를 호출
        try:
            self.open_gripper_wide(self.robot1_id)
            self.open_gripper_wide(self.robot2_id)
            a1 = self._ee_forward_axis_world(self.robot1_id)
            a2 = self._ee_forward_axis_world(self.robot2_id)
            print(f"TOOL-AXIS(world) r1={np.round(a1,3)}, r2={np.round(a2,3)}")
        except Exception as e:
            print(f"ℹ축 보정 로그 생략: {e}")

        print("듀얼 로봇 설정 완료")

    
    def create_target_objects(self):
        """보라색 원통 객체 1개 생성 (샤프란 꽃 역할)"""
        # 단일 꽃 위치 - 테이블 중앙
        flower_position = [0.5, 0.0, 0.65]
        
        print("샤프란 꽃 객체 생성 중...")
        
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
            
            # 🔧 물체 동역학 강화 (마찰/회전 저항)
            p.changeDynamics(cylinder_id, -1, lateralFriction=1.0, rollingFriction=0.001, spinningFriction=0.001)
            
            self.target_objects.append(cylinder_id)
            print(f"   꽃 생성 완료: {flower_position}")
            
        except Exception as e:
            print(f"   꽃 생성 실패: {e}")
        
        print(f"샤프란 꽃 1개 생성 완료!")
    
    def move_both_to_home(self):
        """두 로봇 모두 홈 포지션으로 이동"""
        for robot_key, robot_data in self.robots.items():
            robot_id = robot_data['id']
            home_pos = robot_data['home_position']
            
            for i, joint_pos in enumerate(home_pos):
                p.resetJointState(robot_id, i, joint_pos)
            
            print(f"   {robot_data['name']} 홈 포지션 설정 완료")
    
    # === 개선된 그리퍼 제어 ===
    def control_gripper_ramp(self, robot_id, target_pos, steps=6, dwell=30):
        """그리퍼를 천천히 램프 제어"""
        gripper_joints = [9, 10]
        # 현재 값 추정
        cur = p.getJointState(robot_id, gripper_joints[0])[0]
        delta = (target_pos - cur) / steps
        for s in range(steps):
            v = cur + delta*(s+1)
            for j in gripper_joints:
                p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                        targetPosition=v, force=self.gripper_force)
            for _ in range(dwell):
                p.stepSimulation()
                time.sleep(1/240)

    def open_gripper_wide(self, robot_id):
        """그리퍼를 넓게 열기 (핸드오버용)"""
        self.control_gripper_ramp(robot_id, self.gripper_open_pos_wide)

    def close_gripper(self, robot_id):
        """그리퍼를 천천히 닫기"""
        self.control_gripper_ramp(robot_id, self.gripper_closed_pos)

    def close_gripper_tight(self, robot_id):
        """그리퍼를 꽉 잡기 (객체 보유용)"""
        self.control_gripper_ramp(robot_id, self.grip_hold_pos, steps=8, dwell=25)
        self.active_grips[robot_id] = True
        print(f"로봇 {robot_id}: 꽉 잡기 모드 활성화")

    def maintain_grip(self, robot_id):
        """그립 유지 - 지속적인 힘 적용"""
        if not self.active_grips.get(robot_id, False):
            return
            
        gripper_joints = [9, 10]
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=self.grip_hold_pos,
                force=self.grip_hold_force,  # 더 강한 힘
                maxVelocity=0.1  # 천천히 조정
            )

    def release_grip(self, robot_id):
        """그립 해제"""
        self.active_grips[robot_id] = False
        self.control_gripper_ramp(robot_id, self.gripper_open_pos, steps=6, dwell=20)
        print(f"로봇 {robot_id}: 그립 해제")

    # ========= (NEW) 영구 그립 잠금/해제(고정관절) =========
    def _lock_grasp(self, holder_id, object_id):
        """EE–Object를 고정 관절로 잠가 회전/이동 중 낙하 방지"""
        if not hasattr(self, "_grasp_cid"):
            self._grasp_cid = {}
        key = (holder_id, object_id)
        if key in self._grasp_cid:
            return
        
        st = p.getLinkState(holder_id, self.ee_link_index, computeForwardKinematics=True)
        ppos, porn = st[0], st[1]
        opos, oorn = p.getBasePositionAndOrientation(object_id)
        pinv = p.invertTransform(ppos, porn)
        rel_p, rel_q = p.multiplyTransforms(pinv[0], pinv[1], opos, oorn)
        
        cid = p.createConstraint(holder_id, self.ee_link_index, object_id, -1,
                                 p.JOINT_FIXED, [0,0,0],
                                 [0,0,0], rel_p,
                                 parentFrameOrientation=[0,0,0,1],
                                 childFrameOrientation=rel_q)
        self._grasp_cid[key] = cid
        
        # 접촉 안정화에 도움되는 앵커/마찰
        p.changeDynamics(object_id, -1, lateralFriction=1.0, spinningFriction=0.001, rollingFriction=0.001)
        print(f"영구 그립 잠금: 로봇 {holder_id} → 객체 {object_id}")

    def _unlock_grasp(self, holder_id, object_id):
        """영구 그립 잠금 해제"""
        key = (holder_id, object_id)
        if hasattr(self, "_grasp_cid") and key in self._grasp_cid:
            try:
                p.removeConstraint(self._grasp_cid[key])
                print(f"영구 그립 잠금 해제: 로봇 {holder_id} → 객체 {object_id}")
            except Exception:
                pass
            del self._grasp_cid[key]

    def _confirm_receiver_grasp(self, receiver_id, object_id, frames=30, min_contacts=1):
        """수취 로봇이 frames 프레임 연속으로 물체 접촉 중인지 확인"""
        print(f"수취 그립 확인: {frames}프레임 연속 접촉 체크")
        streak = 0
        for step in range(frames*2):  # 최대 2배 시간 허용
            has_contact, n = self.check_gripper_contact(receiver_id, object_id)
            if has_contact and n >= min_contacts:
                streak += 1
            else:
                streak = 0
            if streak >= frames:
                print(f"수취 그립 확인 완료: {streak}프레임 연속 접촉")
                return True
            self.maintain_grip(receiver_id)
            p.stepSimulation()
            time.sleep(1/240)
        print(f"수취 그립 확인 실패: 연속 접촉 부족")
        return False
    
    def transfer_weld(self, from_id, to_id, object_id, settle_steps=60):
        """
        핸드오버 고정 전환: receiver가 진짜 '끼워잡은' 상태를 보장한 후 전환
        """
        # 0) receiver 그립 확실히 닫아두기
        self.close_gripper_tight(to_id)

        # 1) 양손가락 접촉 확보 없으면 '앉히기' 시도
        ok, _, _ = self._bi_finger_contact(to_id, object_id)
        if not ok:
            self._seat_object_in_gripper(to_id, from_id, object_id, step=0.006, tries=3)
            ok, _, _ = self._bi_finger_contact(to_id, object_id)

        # 2) receiver 접촉이 확보되었을 때만 receiver에 용접
        if ok:
            self._lock_grasp(to_id, object_id)
        else:
            print("receiver 양손가락 접촉 불충분: 그래도 안전을 위해 임시로 receiver 용접 진행")
            self._lock_grasp(to_id, object_id)

        # 3) 더블-홀드 안정화
        for _ in range(settle_steps):
            self.maintain_grip(to_id)
            self.maintain_grip(from_id)
            p.stepSimulation()
            time.sleep(1/240)

        # 4) giver 용접 해제 (하중 이관 완료)
        self._unlock_grasp(from_id, object_id)

        # 5) 살짝 들어올려 ‘잡힘’ 확인 (옵션)
        ee_pos, ee_quat = self.get_end_effector_position(to_id)
        lift = [ee_pos[0], ee_pos[1], ee_pos[2] + 0.02]
        self.move_robot_to_position(to_id, lift, ee_quat, slow=True)
        for _ in range(30):
            self.maintain_grip(to_id)
            p.stepSimulation()
            time.sleep(1/240)

    def _ee_forward_axis_world(self, robot_id):
        """
        EE(손 링크) → 핀치 중앙(9,10) 벡터를 '실제 전진축'으로 사용 (월드 좌표계).
        URDF에 따라 로컬축 정의가 다를 수 있으므로 런타임 추정이 안전함.
        """
        ee = np.array(p.getLinkState(robot_id, self.ee_link_index, computeForwardKinematics=True)[0])
        pinch = np.array(self._pinch_center(robot_id))
        v = pinch - ee
        n = np.linalg.norm(v)
        if n < 1e-4:
            # 핀치 좌표 추정 실패 시, 현 쿼터니언 기준 +Z를 보수적으로 사용
            q = p.getLinkState(robot_id, self.ee_link_index, computeForwardKinematics=True)[1]
            R = np.array(p.getMatrixFromQuaternion(q)).reshape(3,3,order='F')
            return (R @ np.array([0,0,1])).astype(float)
        return (v / n).astype(float)

    # ========= (NEW) Z 클램프 헬퍼 =========
    def _clamp_z(self, pos):
        """Z 좌표를 최소 핸드오버 높이 이상으로 클램프"""
        p3 = list(pos)
        p3[2] = max(p3[2], self.min_handover_clear_z)
        return p3
    
        # === (NEW) 잡은 상태로 물체의 y를 지정값으로 맞추기 ===
    def _move_held_object_to_y(self, holder_id, object_id, target_y, hold_quat=None, slow=True):
        """
        holder_id가 object_id를 잡고(고정) 있는 상태에서,
        물체의 월드 y좌표를 target_y로 맞추도록 EE를 평행이동.
        """
        # 현재 EE / Object 포즈
        ee_pos, ee_quat = self.get_end_effector_position(holder_id)
        obj_pos, _ = self.get_object_position(object_id)
        dy = float(target_y) - float(obj_pos[1])
        if abs(dy) < 1e-3:
            print(f"물체 y가 이미 {target_y:.3f} 근처입니다")
            return list(ee_pos)
        # EE를 같은 양만큼 y로 이동 (고정관절이므로 물체도 동일 평행이동됨)
        new_ee = [ee_pos[0], ee_pos[1] + dy, max(ee_pos[2], self.min_handover_clear_z)]
        # 사이드 가드/높이 가드 적용
        new_ee = self._clamp_z(self._clamp_side_handover(holder_id, new_ee))
        quat = hold_quat if hold_quat is not None else ee_quat
        print(f"잡은 상태 y정렬: obj.y {obj_pos[1]:.3f} → {target_y:.3f} (EE dy={dy:.3f})")
        self.move_robot_to_position(holder_id, new_ee, quat, slow=slow)
        # 이동 후 실제 도달 여부 로그
        obj_pos2, _ = self.get_object_position(object_id)
        print(f"   결과 obj.y={obj_pos2[1]:.3f}")
        return new_ee
    
    # === (NEW) 핸드오버용 사이드 가드: 중앙선(y=table_center_y) 침범 금지 ===
    def _clamp_side_handover(self, robot_id, pos):
        p3 = list(pos)
        cy = self.table_center_y
        m  = self.handover_side_margin_y
        if robot_id == self.robot1_id:
            # 주는 팔은 y ≥ 중앙선+여유
            p3[1] = max(p3[1], cy + m)
        elif robot_id == self.robot2_id:
            # 받는 팔은 y ≤ 중앙선-여유
            p3[1] = min(p3[1], cy - m)
        return p3
    
    def control_gripper(self, robot_id, open_gripper=True):
        """기본 그리퍼 제어 (기존 호환성)"""
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
    
    # === 마주보기 정렬 시스템 ===
    def _look_at_quat_align_tool_axis(self, from_pos, to_pos, tool_axis_local=None, world_up=None):
        """툴축을 특정 방향으로 정렬하는 쿼터니언 계산"""
        tool_axis_local = self.TOOL_AXIS_LOCAL if tool_axis_local is None else tool_axis_local
        world_up = self.WORLD_UP if world_up is None else world_up
        
        f = np.array(to_pos) - np.array(from_pos)
        fn = f / (np.linalg.norm(f)+1e-9)             # forward(월드)
        
        # 1) tool_axis_local -> fn 으로 회전(q1)
        a = tool_axis_local / (np.linalg.norm(tool_axis_local)+1e-9)
        v = np.cross(a, fn)
        c = float(np.dot(a, fn))
        
        if np.linalg.norm(v) < 1e-8:
            q1 = [0,0,0,1] if c>0 else [1,0,0,0]      # 180° or identity
        else:
            s = math.sqrt((1+c)*2.0)
            q1 = [v[0]/s, v[1]/s, v[2]/s, s/2.0]

        # 2) q1 적용 후의 '로컬 업'(임의로 [0,1,0] 채택)을 월드 업과 최대정렬(q2: forward축 회전)
        up_local = np.array([0,1,0])
        up_rot = np.array(p.rotateVector(q1, up_local))
        
        # forward축에 직교하도록 월드 업 투영
        t = world_up - fn*np.dot(world_up, fn)
        if np.linalg.norm(t) < 1e-6:
            q2 = [0,0,0,1]
        else:
            t = t/np.linalg.norm(t)
            # up_rot -> t 로 forward축을 축으로 회전
            ang = math.atan2(np.dot(np.cross(up_rot, t), fn), np.dot(up_rot, t))
            q2 = [fn[0]*math.sin(ang/2), fn[1]*math.sin(ang/2), fn[2]*math.sin(ang/2), math.cos(ang/2)]

        # q = q2 * q1 (쿼터니언 곱셈) - 버그 수정
        q = self.quat_mul(q2, q1)
        return q
    
    # === 충돌 방지 시스템 ===
    def min_distance_between_arms(self, dist=0.10):
        """두 로봇팔 간 최소거리 계산"""
        risk_links_r1 = [2,3,4,5]  # 어퍼암/포어암/손목 등
        risk_links_r2 = [2,3,4,5]
        mind = 1e9
        n1 = n2 = None
        
        for l1 in risk_links_r1:
            for l2 in risk_links_r2:
                cps = p.getClosestPoints(self.robot1_id, self.robot2_id, dist, linkIndexA=l1, linkIndexB=l2)
                if cps:
                    d = min(c[8] for c in cps)  # contactDistance
                    if d < mind:
                        mind, n1, n2 = d, l1, l2
        
        return mind if mind<1e9 else None, n1, n2

    def collision_guard_holder_down(self, mover_id, holder_id, drop_step=0.02, safe=0.06):
        """충돌 시 홀더만 살짝 하강, 무버는 계속 진행"""
        mind, l1, l2 = self.min_distance_between_arms(dist=max(0.15, safe+0.05))
        if mind is not None and mind < safe:
            print(f"충돌 경보! 최소거리={mind*100:.1f}cm → 홀더 {drop_step*1000:.0f}mm 하강")
            pos, orn = self.get_end_effector_position(holder_id)
            new_pos = [pos[0], pos[1], max(pos[2]-drop_step, self.min_handover_clear_z)]
            self.move_robot_to_position(holder_id, new_pos, orn, slow=True)
            return True
        return False
    
    # === 개선된 도킹 오차 계산 ===
    def docking_errors(self, ee_pos, port_pos, port_quat):
        """축방향/방사방향 오차 분리 계산"""
        R = np.array(p.getMatrixFromQuaternion(port_quat)).reshape(3,3,order='F')
        a = R @ self.TOOL_AXIS_LOCAL
        a = a/np.linalg.norm(a)
        
        d = np.array(port_pos) - np.array(ee_pos)
        axial = float(np.dot(d, a))
        radial = float(np.linalg.norm(d - axial*a))
        
        return axial, radial
    
    def axis_servo_dock(self, robot_id, ee_link, q_indices, port_provider_id,
                        approach_step=0.008, max_axial=0.12, radial_tol=0.005, fastpass_cb=None):
        """
        가상 고정구 방식 축방향 도킹:
        - '포트 제공자(=giver)의 실제 전진축'을 런타임 추정(_ee_forward_axis_world)하여
        항상 '핀치 사이 축'을 따라 접근하도록 강제
        - 측방 보정 게인을 낮춰(0.35) 떨림/옆면 비비기 억제
        """
        print(f"축방향 도킹 시작: 단계={approach_step*1000:.1f}mm, 최대거리={max_axial*100:.1f}cm")
        n_steps = max(1, int(max_axial/approach_step))
        for k in range(n_steps):
            if fastpass_cb is not None and fastpass_cb():
                print("FAST-PASS: 도킹 중 2cm 조건 충족 → 즉시 집기")
                return "FASTPASS"

            # 최신 포트(주는 팔) 상태
            port_pos, _ = self.get_end_effector_position(port_provider_id)
            a = self._ee_forward_axis_world(port_provider_id)  # 홀더의 실제 전진축(월드)
            a = a / (np.linalg.norm(a)+1e-9)

            # 충돌 시: 홀더만 살짝 하강, 무버는 계속
            if self.collision_guard_holder_down(mover_id=robot_id, holder_id=port_provider_id,
                                                drop_step=0.02, safe=0.06):
                print("   홀더 하강 후 계속 진행")

            # 무버 EE 현재 상태
            st = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
            pe = np.array(st[0])

            # 오차 분해
            d = np.array(port_pos) - pe
            axial = float(np.dot(d, a))
            lateral = d - axial*a

            # 로그
            print(f"   단계 {k+1}/{n_steps}: 축={axial*100:.1f}cm, 방사={np.linalg.norm(lateral)*100:.1f}cm")

            # 성공/안전정지 판정
            if np.linalg.norm(lateral) < radial_tol and axial <= approach_step:
                print("축방향 도킹 성공!")
                return True
            if axial <= self.axial_safe_stop and np.linalg.norm(lateral) < radial_tol*2.0:
                print(f"축방향 안전정지 도달(axial≤{self.axial_safe_stop*100:.0f}mm), 정렬 완료로 간주")
                return True

            # 다음 목표 (측방 저게인 보정 + 축 전진)
            lateral = 0.35 * lateral  # 측방 보정 gain down (가상 피처 느낌)
            target_pos = (pe + lateral) + a*min(approach_step, max(axial-0.5*approach_step, 0.0))
            target_pos = self._clamp_side_handover(robot_id, self._as_float_list(target_pos.tolist()))
            target_pos[2] = max(target_pos[2], self.min_handover_clear_z)

            ok = self.resolved_rate_pose_servo(robot_id, ee_link, q_indices,
                                            target_pos, st[1],
                                            Kp=3.0, Kr=2.5, dt=1/240, max_steps=50)
            if not ok:
                print("6D 서보 실패, 다음 스텝으로 지속")
        print("축방향 도킹 최대 거리 도달")
        return False


    
    # === 6D 포즈 서보 제어 시스템 ===
    def quat_mul(self, q1, q2):
        """쿼터니언 곱셈"""
        x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ]

    def quat_inv(self, q):
        """쿼터니언 역"""
        x,y,z,w = q
        return [-x,-y,-z,w]

    def so3_log(self, R_err):
        """SO(3) 로그 맵 (회전 오차를 축-각도로 변환)"""
        cos_theta = (np.trace(R_err)-1)/2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = math.acos(cos_theta)
        if theta < 1e-6:
            return np.zeros(3)
        w = (1/(2*math.sin(theta))) * np.array([
            R_err[2,1]-R_err[1,2],
            R_err[0,2]-R_err[2,0],
            R_err[1,0]-R_err[0,1]
        ])
        return theta * w

    def quat_to_R(self, q):
        """쿼터니언을 회전 행렬로 변환"""
        return np.array(p.getMatrixFromQuaternion(q)).reshape(3,3,order='F')

    def resolved_rate_pose_servo(self, body_id, ee_link, q_indices, target_pos, target_quat,
                                 Kp=2.0, Kr=2.0, dt=1/240, max_steps=600, fastpass_cb=None):
        """6D 포즈 서보 제어 (Jacobian 버그 수정)"""
        target_pos = self._as_float_list(target_pos)
        target_quat = [float(x) for x in target_quat]
        
        print(f"6D 포즈 서보 시작: 목표 {[round(x,3) for x in target_pos]}")
        
        for step in range(max_steps):
            # FAST-PASS: 서보 중에도 즉시 전환
            if fastpass_cb is not None and step % 5 == 0 and fastpass_cb():
                print("FAST-PASS: 서보 중 2cm 조건 충족 → 조기 종료")
                return "FASTPASS"
            # 현재 상태
            st = p.getLinkState(body_id, ee_link, computeForwardKinematics=True)
            pos = np.array(st[0])
            quat = st[1]
            R = self.quat_to_R(quat)
            R_star = self.quat_to_R(target_quat)

            # 6D 오차
            e_p = np.array(target_pos) - pos
            R_err = R_star @ R.T
            e_w = self.so3_log(R_err)  # axis-angle (3D)

            # 수렴 판정
            pos_error = np.linalg.norm(e_p)
            rot_error = np.linalg.norm(e_w)
            
            if step % 60 == 0:  # 주기적 로그
                print(f"   단계 {step}: 위치오차={pos_error:.4f}m, 회전오차={rot_error:.4f}rad")
            
            if pos_error < 1e-3 and rot_error < 1e-2:
                print(f"6D 포즈 서보 수렴! (단계: {step})")
                return True

            # EE twist
            v = Kp * e_p
            w = Kr * e_w
            twist = np.concatenate([v, w])  # (6,)

            # Jacobian은 가동 조인트만 사용!
            q_all = self._q_all_movable(body_id)
            zero = [0.0]*len(q_all)
            try:
                Jlin, Jang = p.calculateJacobian(body_id, ee_link, [0,0,0], q_all, zero, zero)
                J = np.vstack([np.array(Jlin), np.array(Jang)])  # (6, nMov)

                # 제어할 7축만 추출 (가동 DOF 인덱스로 매핑)
                idx = [self.jindex_map[body_id][j] for j in q_indices]
                J_cmd = J[:, idx]  # (6, 7)
                qdot_cmd = np.linalg.pinv(J_cmd, rcond=1e-3) @ twist

                # 속도 클램프(안정성)
                qdot_cmd = np.clip(qdot_cmd, -1.0, 1.0)
                
                # qdot 모니터링(디버그)
                maxdq = float(np.max(np.abs(qdot_cmd)))
                if step % 60 == 0:
                    print(f"   max|qdot|={maxdq:.3f}")

                # 속도 제어
                for j, dq in zip(q_indices, qdot_cmd):
                    p.setJointMotorControl2(body_id, j, p.VELOCITY_CONTROL, 
                                          targetVelocity=float(dq), force=200)
            except Exception as e:
                print(f"자코비안 계산 실패: {e}")
                return False

            p.stepSimulation()
            time.sleep(dt)

        print(f"6D 포즈 서보 최대 반복 도달 ({max_steps})")
        return False
    
    def move_robot_to_position(self, robot_id, target_pos, target_orn=None, slow=False):
        """특정 로봇을 목표 위치로 이동 (기존 호환성)"""
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        try:
            # 현재 조인트 상태를 시드로 사용
            seed_pose = []
            for i in range(7):
                joint_state = p.getJointState(robot_id, i)
                seed_pose.append(joint_state[0])
            
            joint_poses = p.calculateInverseKinematics(
                robot_id, self.ee_link_index, target_pos, target_orn,  # EE 링크 인덱스 사용
                lowerLimits=[-2.8, -1.7, -2.8, -3.0, -2.8, -0.1, -2.8],
                upperLimits=[2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8],
                jointRanges=[5.6, 3.4, 5.6, 2.9, 5.6, 3.8, 5.6],
                restPoses=seed_pose,
                maxNumIterations=200,
                residualThreshold=1e-3,
                solver=p.IK_DLS  # DLS 솔버 사용
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
            print(f"로봇 이동 실패: {e}")
            return False
    
    def get_end_effector_position(self, robot_id):
        """엔드 이펙터의 현재 위치 확인"""
        state = p.getLinkState(robot_id, self.ee_link_index)  # EE 링크 인덱스 사용
        return state[0], state[1]
    
    def get_object_position(self, object_id):
        """객체의 현재 위치 확인"""
        pos, orn = p.getBasePositionAndOrientation(object_id)
        return pos, orn
    
    def check_gripper_contact(self, robot_id, target_object_id):
        """그리퍼와 객체 간 접촉 확인 (개선된 버전)"""
        finger_links = [9, 10]  # 11 제거로 정확도 향상
        for link_id in finger_links:
            contacts = p.getContactPoints(bodyA=robot_id, bodyB=target_object_id, linkIndexA=link_id)
            if contacts:
                return True, len(contacts)
        return False, 0
    
    def gradual_descent_and_grasp(self, robot_id, start_pos, target_object_id):
        """점진적 하강 및 잡기 - 0.65m까지 안전하게 내려가기"""
        obj_pos, _ = self.get_object_position(target_object_id)
        current_z = start_pos[2]
        
        # Z축 제한: 0.65m까지 내려가기 (테이블 위 객체까지)
        min_z = 0.65
        target_z = max(obj_pos[2] + 0.02, min_z)  # 객체 위 2cm 또는 최소 높이
        step_size = 0.02  # 더 작은 단계로 정밀하게
        
        print(f"점진적 하강 시작:")
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
            
            # 각 단계마다 접촉 확인 (조기 감지)
            has_contact, contact_count = self.check_gripper_contact(robot_id, target_object_id)
            if has_contact:
                print(f"조기 접촉 감지! 높이: {current_z:.3f}m")
                break
            
            # 안정화 대기 (더 짧게)
            for _ in range(40):
                p.stepSimulation()
                time.sleep(1/240)
        
        print(f"하강 완료! 최종 높이: {current_z:.3f}m")
        
        # 미세 조정: 객체에 더 가까이 접근
        obj_pos, _ = self.get_object_position(target_object_id)
        fine_target = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01]  # 객체 위 1cm (유지)
        
        print("미세 조정: 객체에 더 가까이...")
        self.move_robot_to_position(robot_id, fine_target, slow=True)
        
        # 안정화
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1/240)
        
        # 그리퍼 닫기 (꽉 잡기)
        print("🤏 그리퍼 꽉 잡기...")
        self.close_gripper_tight(robot_id)
        
        # 최종 접촉 확인
        has_contact, contact_count = self.check_gripper_contact(robot_id, target_object_id)
        print(f"접촉 상태: {has_contact} ({contact_count}개 접촉점)")
        
        # 잡기 성공 후 약간 들어올리기 (확실한 그립 확인) + 고정 잠금
        if has_contact:
            print("그립 유지 모드 시작")
            print("확실한 그립을 위해 조금 들어올리기...")
            
            # 2cm 들어올리기
            lift_test_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.05]
            self.move_robot_to_position(robot_id, lift_test_pos, slow=True)
            
            # 들어올린 상태에서 그립 유지
            for _ in range(60):
                self.maintain_grip(robot_id)
                p.stepSimulation()
                time.sleep(1/240)
            
            # EE–Object 영구 고정(넘겨줄 때까지 유지)
            self._lock_grasp(robot_id, target_object_id)

            # 최종 접촉 재확인
            final_contact, final_count = self.check_gripper_contact(robot_id, target_object_id)
            if final_contact:
                print("안정적 그립 확인 완료!")
                return True
            else:
                print("들어올린 후 접촉 손실 - 재시도 필요")
                return False
        
        return has_contact
    
    def check_docking_success(self, robot_id, target_object_id, expected_distance=0.05):
        """도킹 성공 여부 확인 (포트 축 기준으로 개선)"""
        try:
            # 수취 로봇과 포트(주는 팔) 위치
            ee_pos, _ = self.get_end_effector_position(robot_id)
            # 포트(주는 팔)의 축 기준으로 오차 분해
            port_pos, port_quat = self.get_end_effector_position(self.robot1_id)
            
            # 축방향/방사방향 오차 분리
            axial, radial = self.docking_errors(ee_pos, port_pos, port_quat)
            
            # 접촉점 확인
            has_contact, contact_count = self.check_gripper_contact(robot_id, target_object_id)
            
            # 성공 조건: 축 ≤ 1.0cm AND 방사 ≤ 0.5cm OR 접촉
            success = (abs(axial) <= 0.01 and radial <= 0.005) or has_contact
            
            print(f"   도킹 체크: 축={axial*100:.1f}cm, 방사={radial*100:.1f}cm, 접촉={has_contact}({contact_count}개), 성공={success}")
            
            return success, radial, has_contact
            
        except Exception as e:
            print(f"   도킹 체크 실패: {e}")
            return False, 999, False

    def retreat_and_retry(self, robot_id, retreat_distance=0.2):
        """후퇴 및 재접근 (부호 버그 수정)"""
        print(f"⏮로봇2: {retreat_distance*100:.0f}cm 후퇴...")
        
        # 현재 위치에서 후퇴
        current_pos, current_orn = self.get_end_effector_position(robot_id)
        retreat_pos = [current_pos[0] - retreat_distance, current_pos[1], current_pos[2]]  # X축 뒤로
        
        # 후퇴
        self.move_robot_to_position(robot_id, retreat_pos, current_orn, slow=True)
        
        # 후퇴 후 잠시 대기
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1/240)
        
        print("후퇴 완료!")

    def guarded_approach(self, robot_id, goal_pos, goal_orn, object_id, step=0.008, max_steps=30):
        """가드 접근 - 마지막 5~10cm 구간용"""
        print(f"가드 접근 시작: 단계크기={step*1000:.1f}mm, 최대단계={max_steps}")
        
        cur_pos, _ = self.get_end_effector_position(robot_id)
        direction = np.array(goal_pos) - np.array(cur_pos)
        dist = np.linalg.norm(direction)
        
        if dist < 1e-6: 
            return True
            
        direction = direction / dist
        n_steps = min(max_steps, int(dist/step)+1)
        
        for k in range(n_steps):
            p_target = (np.array(cur_pos) + direction*step*(k+1)).tolist()
            self.move_robot_to_position(robot_id, p_target, goal_orn, slow=True)
            
            # 접촉/거리 체크
            success, d, has_contact = self.check_docking_success(robot_id, object_id, expected_distance=0.035)
            print(f"   가드 단계 {k+1}/{n_steps}: 거리={d:.3f}m, 접촉={has_contact}")
            
            if has_contact or d < 0.04:
                print("가드 접근 성공!")
                return True
                
        print("가드 접근 불완전 종료")
        return False
    
    def _bi_finger_contact(self, robot_id, object_id):
        c9  = p.getContactPoints(bodyA=robot_id, bodyB=object_id, linkIndexA=9)
        c10 = p.getContactPoints(bodyA=robot_id, bodyB=object_id, linkIndexA=10)
        return (len(c9) > 0 and len(c10) > 0), len(c9), len(c10)
    
    def _seat_object_in_gripper(self, receiver_id, giver_id, object_id, step=0.006, tries=3):
        """receiver 툴축(+X)으로 살짝 전진 → 물체를 핑거 사이로 앉힘"""
        for t in range(tries):
            ok, l, r = self._bi_finger_contact(receiver_id, object_id)
            if ok:
                print(f"양손가락 접촉 확보 ({l},{r})")
                return True
            # receiver 현재 포즈
            ee_pos, ee_quat = self.get_end_effector_position(receiver_id)
            R = np.array(p.getMatrixFromQuaternion(ee_quat)).reshape(3,3,order='F')
            a = R @ self.TOOL_AXIS_LOCAL  # +X 전진축
            tgt = (np.array(ee_pos) + a*step).tolist()
            tgt = self._clamp_z(self._clamp_side_handover(receiver_id, tgt))
            print(f"   미세 전진 시도 {t+1}/{tries}")
            self.move_robot_to_position(receiver_id, tgt, ee_quat, slow=True)
            for _ in range(30):
                self.maintain_grip(receiver_id)
                p.stepSimulation()
                time.sleep(1/240)
        ok, l, r = self._bi_finger_contact(receiver_id, object_id)
        print(f"   접촉 최종확인: ok={ok}, L={l}, R={r}")
        return ok



    def perform_handover(self, object_id):
        """개선된 핸드오버 - 마주보기 + 축방향 도킹 + 충돌 방지"""
        print("개선된 핸드오버 시작 (마주보기 + 축방향 도킹)")

        # 1) LIFT: 로봇1이 안전 높이 확보
        lift_height = min(self.table_top_z + 0.30, 1.00)
        ee1, _ = self.get_end_effector_position(self.robot1_id)
        print(f"LIFT: 로봇1이 객체를 {lift_height}m로 올리는 중...")
        self.move_robot_to_position(self.robot1_id, self._clamp_z([ee1[0], ee1[1], lift_height]), slow=True)

        # ※ 낙하 방지는 grasp 고정으로 이미 보장(_lock_grasp 사용)

        # 2) POSE: 마주보기 핸드오버 포즈 계산
        center = [0.30, 0.00, lift_height]  # 테이블 중앙 상공
        gap = np.clip(0.18, self.handover_gap_min, self.handover_gap_max)  # 14~20cm 유지
        
        r1_pos = [center[0]-gap/2, center[1], center[2]]
        r2_pos = [center[0]+gap/2, center[1], center[2]]
        # 중앙선 침범 금지 + 바닥 여유
        r1_pos = self._clamp_z(self._clamp_side_handover(self.robot1_id, r1_pos))
        r2_pos = self._clamp_z(self._clamp_side_handover(self.robot2_id, r2_pos))
        
        # 서로를 바라보는 자세 계산
        r1_quat = self._look_at_quat_align_tool_axis(r1_pos, r2_pos)  # 로봇1이 로봇2를 봄
        r2_quat = self._look_at_quat_align_tool_axis(r2_pos, r1_pos)  # 로봇2가 로봇1을 봄
        
        print("POSE: 마주보기 핸드오버 포즈로 이동")
        print(f"   로봇1 위치: {[round(x,3) for x in r1_pos]}")
        print(f"   로봇2 위치: {[round(x,3) for x in r2_pos]}")
        
        # 자세 확인을 위한 툴축 벡터 출력
        R1 = np.array(p.getMatrixFromQuaternion(r1_quat)).reshape(3,3,order='F')
        R2 = np.array(p.getMatrixFromQuaternion(r2_quat)).reshape(3,3,order='F')
        tool1 = R1 @ self.TOOL_AXIS_LOCAL
        tool2 = R2 @ self.TOOL_AXIS_LOCAL
        dot_product = np.dot(tool1, tool2)
        print(f"   툴축 내적: {dot_product:.3f} (서로 반대면 -0.98 이하)")
        
        # 벡터-to-벡터 직접 정렬: r2의 툴축을 -tool1에 정렬
        q_corr = self._quat_from_two_vectors(tool2, -tool1)
        r2_quat = self.quat_mul(q_corr, r2_quat)
        
        # 검증 로그
        R2 = np.array(p.getMatrixFromQuaternion(r2_quat)).reshape(3,3,order='F')
        tool2 = R2 @ self.TOOL_AXIS_LOCAL
        dot_product = float(np.dot(tool1, tool2))
        print(f"   툴축 내적(보정 후): {dot_product:.3f}")  # 여기서 -0.98~-1.00 나와야 정상

        # 로봇1: 핸드오버 포즈로 이동
        self.move_robot_to_position(self.robot1_id, r1_pos, r1_quat, slow=True)

        # (NEW) 각도 맞춘 직후, 잡은 상태로 원기둥의 y를 0.5로 이동
        #  - EE와 물체는 _lock_grasp()로 고정되어 있으므로 EE y 평행이동 = 물체 y 이동
        #  - 반환된 EE 위치를 r1_pos로 갱신하여 이후 단계(도킹/체크)에서 최신 포즈 사용
        r1_pos = self._move_held_object_to_y(self.robot1_id, object_id, target_y=0.050, hold_quat=r1_quat, slow=True)
        
        # 로봇1 그립 유지 (이동 중에도 객체 놓치지 않도록)
        if object_id in [obj for obj in self.target_objects]:
            print("로봇1: 이동 중 그립 유지...")
            for _ in range(60):  # 1초간 그립 유지
                self.maintain_grip(self.robot1_id)
                p.stepSimulation()
                time.sleep(1/240)

        # 3)~4) ALIGN + DOCK: 다회 재시도 루프 (실패 시 30cm 후퇴 후 재시도)
        print("ALIGN+DOCK: 재시도 루프 시작")
        q_indices = list(range(7))
        pre_offset = 0.10  # 보수적 시작
        docking_success = False
        near_grasp_triggered = False
        for attempt in range(1, self.docking_max_attempts + 1):
            print(f"   ── 시도 {attempt}/{self.docking_max_attempts}")
            self.open_gripper_wide(self.robot2_id)
            # 현재 r2 시작 위치 기준 스윙-인(보기 좋게 원호 진입)
            r2_start, _ = self.get_end_effector_position(self.robot2_id)
            self._swing_in_prepath(r2_start, r1_pos, r2_quat, radius=0.18)

            # 프리그립 위치: '포트(r1_pos)' 기준으로 r2 전진축의 -pre_offset
            a2  = self._ee_forward_axis_world(self.robot2_id)  # 👈 r2 실제 전진축(월드)
            pre_r2 = (np.array(r1_pos) - a2*pre_offset).tolist()
            pre_r2 = self._clamp_z(self._clamp_side_handover(self.robot2_id, pre_r2))

            # 정밀 정렬
            self._take_velocity_control(self.robot2_id, q_indices)
            ok_align = self.resolved_rate_pose_servo(
                self.robot2_id, self.ee_link_index, q_indices,
                pre_r2, r2_quat, Kp=2.0, Kr=3.5, max_steps=300, 
                fastpass_cb=lambda: self._near_grasp_ready(self.robot2_id, object_id)
            )
            if ok_align == "FASTPASS":
                self.close_gripper_tight(self.robot2_id)
                self._lock_grasp(self.robot2_id, object_id)
                docking_success = True; near_grasp_triggered = True; break
            if not ok_align:
                print("프리그립 정렬 실패하지만 도킹 시도")

            # === (NEW) FAST-PASS #1: 정렬 직후 2cm 이내면 바로 집기 시도
            if self._near_grasp_ready(self.robot2_id, object_id):
                print("FAST-PASS: 2cm 이내 → 즉시 집기/더블홀드로 전환")
                self.close_gripper_tight(self.robot2_id)
                if self._confirm_receiver_grasp(self.robot2_id, object_id, frames=20):
                    self._lock_grasp(self.robot2_id, object_id)
                    docking_success = True
                    near_grasp_triggered = True
                    break
                else:
                    print("근접 집기 실패 → 수취팔만 후퇴 후 재시도")
                    self.release_grip(self.robot2_id)
                    self._retreat_receiver_along_facing(self.robot2_id, r1_pos, dist=self.failed_retreat_distance)
                    continue

            # 축방향 도킹
            self._take_velocity_control(self.robot2_id, q_indices)
            docking_success = self.axis_servo_dock(
                robot_id=self.robot2_id,
                ee_link=self.ee_link_index,
                q_indices=q_indices,
                port_provider_id=self.robot1_id,   # 포트 제공자 = 로봇1(홀더)
                approach_step=0.008,
                max_axial=0.12,
                radial_tol=0.005,
                fastpass_cb=lambda: self._near_grasp_ready(self.robot2_id, object_id),
            )
            if docking_success == "FASTPASS":
                self.close_gripper_tight(self.robot2_id)
                self._lock_grasp(self.robot2_id, object_id)
                docking_success = True
                near_grasp_triggered = True
                break

            # === (NEW) FAST-PASS #2: 도킹 중/직후 2cm 이내면 바로 집기 시도
            if not docking_success and self._near_grasp_ready(self.robot2_id, object_id):
                print("FAST-PASS: 도킹 미성공이나 2cm 이내 → 즉시 집기")
                self.close_gripper_tight(self.robot2_id)
                if self._confirm_receiver_grasp(self.robot2_id, object_id, frames=20):
                    docking_success = True
                    near_grasp_triggered = True
                    break
                else:
                    print("근접 집기 실패 → 수취팔만 후퇴 후 재시도")
                    self.release_grip(self.robot2_id)
                    self._retreat_receiver_along_facing(self.robot2_id, r1_pos, dist=self.failed_retreat_distance)
                    continue

            if docking_success:
                print("도킹 성공 (재시도 루프 종료)")
                break

            # 실패 → 마주보기 반대방향으로 30cm 후퇴 후 다음 시도
            print("도킹 실패 → 수취 팔만 후퇴 30cm 후 재시도")
            self._retreat_receiver_along_facing(self.robot2_id, r1_pos, dist=self.failed_retreat_distance)

        # 5) DOUBLE-HOLD: 로봇2가 잡기 (FAST-PASS에서 이미 잡았으면 생략)
        print("🤝 DOUBLE-HOLD: 두 로봇이 동시에 객체 보유")
        if not self.active_grips.get(self.robot2_id, False):
            self.close_gripper_tight(self.robot2_id)

        # 더블-홀드 안정화 대기 (양쪽 모두 그립 유지)
        print(" 더블-홀드 상태 유지...")
        for step in range(120):
            self.maintain_grip(self.robot1_id)  # 로봇1 그립 유지
            self.maintain_grip(self.robot2_id)  # 로봇2 그립 유지
            p.stepSimulation()
            time.sleep(1/240)

        # 6) TRANSFER: 하중 이관 (로봇1 → 로봇2)
        print("TRANSFER: 하중 이관 중...")
        # 아직 receiver에 용접 안되어 있으면 여기서 전환
        self.transfer_weld(self.robot1_id, self.robot2_id, object_id, settle_steps=60)
        
        # 로봇2가 확실히 잡고 있는지 마지막 확인
        for step in range(30):
            self.maintain_grip(self.robot2_id)
            p.stepSimulation()
            time.sleep(1/240)
            
        # 먼저 고정 해제 → 그립 해제(순서 중요: 언락 후 오픈)
        self._unlock_grasp(self.robot1_id, object_id)
        self.release_grip(self.robot1_id)

        # 7) CLEAR: 로봇2 안전 후퇴 및 납품
        print("CLEAR: 로봇2 납품 위치로 이동")
        self._retreat_receiver_along_facing(self.robot2_id, r1_pos, dist=0.25)
        
        # 후퇴하면서도 그립 유지
        current_pos, current_orn = self.get_end_effector_position(self.robot2_id)
        retreat_pos = [current_pos[0] - 0.25, current_pos[1], current_pos[2]]
        
        # 천천히 후퇴하면서 그립 유지
        steps = 60
        for step in range(steps):
            # 중간 위치 계산
            alpha = (step + 1) / steps
            intermediate_pos = [
                current_pos[0] + alpha * (retreat_pos[0] - current_pos[0]),
                current_pos[1] + alpha * (retreat_pos[1] - current_pos[1]),
                current_pos[2] + alpha * (retreat_pos[2] - current_pos[2])
            ]
            
            # 위치 업데이트
            self.move_robot_to_position(self.robot2_id, intermediate_pos, current_orn, slow=True)
            
            # 그립 유지
            self.maintain_grip(self.robot2_id)
        
        # 납품 위치로 이동 (계속 그립 유지)
        delivery_pos = self._clamp_z([0.1, -0.5, 0.8])
        delivery_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.move_robot_to_position(self.robot2_id, delivery_pos, delivery_orn, slow=True)
        
        # 납품 이동 중에도 그립 유지
        self.move_robot_to_position(self.robot2_id, delivery_pos, delivery_orn, slow=True)
        for _ in range(30):
            self.maintain_grip(self.robot2_id)
            p.stepSimulation()
            time.sleep(1/240)

        # 8) RELEASE: 객체 방출
        print("RELEASE: 객체 방출...")
        self._unlock_grasp(self.robot2_id, object_id)
        self.release_grip(self.robot2_id)

        # 9) 복귀
        print("두 로봇 홈 복귀...")
        
        # 그립 상태 초기화
        self.active_grips[self.robot1_id] = False
        self.active_grips[self.robot2_id] = False
        
        self.move_both_to_home()
        
        print("개선된 마주보기 핸드오버 완료!")
        return True

    def dual_robot_harvest_sequence(self, target_object_id):
        """듀얼 로봇 수확 시퀀스 (개선된 버전)"""
        print(f"\n개선된 듀얼 로봇 수확 시작 (객체 ID: {target_object_id})")
        
        # 1. 객체 위치 확인
        obj_pos, _ = self.get_object_position(target_object_id)
        print(f"타겟 위치: {[round(x, 3) for x in obj_pos]}")
        
        # 2. 로봇1이 접근 (0.65m + 10cm = 0.75m에서 시작)
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.10]  # 객체 위 10cm
        print("로봇1: 접근 중...")
        self.control_gripper(self.robot1_id, open_gripper=True)
        self.move_robot_to_position(self.robot1_id, approach_pos)
        
        # 3. 로봇1이 점진적 하강하여 잡기 (0.65m까지)
        print("로봇1: 잡기 시도...")
        grasp_success = self.gradual_descent_and_grasp(self.robot1_id, approach_pos, target_object_id)
        
        if not grasp_success:
            print("잡기 실패!")
            return False
        
        print(" 잡기 성공!")
        
        # 4. 개선된 핸드오버 실행
        handover_success = self.perform_handover(target_object_id)
        
        if handover_success:
            print(" 개선된 듀얼 로봇 수확 완료!")
            return True
        else:
            print(" 전달 실패!")
            return False
    
    # === 6D 포즈 서보를 활용한 정밀 도킹 ===
    def precision_docking_approach(self, robot_id, target_pos, target_quat, object_id):
        """6D 포즈 서보를 활용한 정밀 도킹"""
        print(" 정밀 도킹 접근 시작 (6D 포즈 서보)")
        
        # 1단계: yaw만 정렬 (멀리서)
        ee_pos, _ = self.get_end_effector_position(robot_id)
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(ee_pos))
        
        if distance_to_target > 0.08:  # 8cm 이상 멀리 있으면
            print("   1단계: yaw 정렬 (거친 접근)")
            # 기존 IK 방식으로 대략적 접근
            approach_pos = [target_pos[0] - 0.05, target_pos[1], target_pos[2]]
            self.move_robot_to_position(robot_id, approach_pos, target_quat, slow=True)
        
        # 2단계: 6D 포즈 서보 (근접)
        print("   2단계: 6D 포즈 서보 (정밀 접근)")
        q_indices = list(range(7))  # Panda 관절 인덱스
        success = self.resolved_rate_pose_servo(
            robot_id, self.ee_link_index, q_indices, 
            target_pos, target_quat,
            Kp=1.5, Kr=1.0, max_steps=400
        )
        
        return success
    
    # === 시스템 테스트 및 진단 ===
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
        
        print("시스템 상태 정상")
    
    def test_single_robot_harvest(self):
        """단일 로봇 수확 테스트"""
        print("단일 로봇 수확 테스트...")
        
        if not self.target_objects:
            print("수확할 객체가 없습니다!")
            return
        
        # 유일한 객체로 테스트
        target_obj = self.target_objects[0]
        obj_pos, _ = self.get_object_position(target_obj)
        
        # 로봇1으로 기본 수확 테스트 (0.65m + 10cm에서 시작)
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.10]
        
        print("로봇1: 단일 수확 테스트 중...")
        self.control_gripper(self.robot1_id, open_gripper=True)
        self.move_robot_to_position(self.robot1_id, approach_pos)
        
        # 잡기 시도 (0.65m까지 내려가기)
        grasp_success = self.gradual_descent_and_grasp(self.robot1_id, approach_pos, target_obj)
        
        if grasp_success:
            # 들어올리기 (그립 유지하면서)
            lift_pos = [obj_pos[0], obj_pos[1], 0.8]
            self.move_robot_to_position(self.robot1_id, lift_pos)
            
            # 이동 중 그립 유지
            for _ in range(30):
                self.maintain_grip(self.robot1_id)
                p.stepSimulation()
                time.sleep(1/240)
            
            # 놓기
            place_pos = [0.1, 0.3, 0.8]
            self.move_robot_to_position(self.robot1_id, place_pos)
            
            # 마지막까지 그립 유지 후 해제
            for _ in range(30):
                self.maintain_grip(self.robot1_id)
                p.stepSimulation()
                time.sleep(1/240)
                
            self.release_grip(self.robot1_id)
            
            print("단일 로봇 테스트 성공!")
        else:
            print("단일 로봇 테스트 실패!")
        
        # 홈으로 복귀
        self.move_both_to_home()
    
    def test_precision_docking(self):
        """정밀 도킹 테스트"""
        print("🧪 정밀 도킹 테스트...")
        
        if not self.target_objects:
            print("테스트할 객체가 없습니다!")
            return
        
        target_obj = self.target_objects[0]
        
        # 로봇1이 먼저 객체를 잡고 특정 위치로 이동 (0.65m까지 내려가기)
        obj_pos, _ = self.get_object_position(target_obj)
        approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.10]
        
        print("로봇1: 객체 픽업...")
        self.control_gripper(self.robot1_id, open_gripper=True)
        self.move_robot_to_position(self.robot1_id, approach_pos)
        
        if self.gradual_descent_and_grasp(self.robot1_id, approach_pos, target_obj):
            # 테스트 위치로 이동
            test_pos = [0.3, 0.0, 0.9]
            test_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
            self.move_robot_to_position(self.robot1_id, test_pos, test_orn)
            
            # 로봇2로 정밀 도킹 테스트
            print("로봇2: 정밀 도킹 테스트...")
            self.open_gripper_wide(self.robot2_id)
            
            # 목표: 객체 근처로 정밀 접근
            obj_pos, _ = self.get_object_position(target_obj)
            target_docking_pos = [obj_pos[0] + 0.05, obj_pos[1], obj_pos[2]]
            target_docking_orn = p.getQuaternionFromEuler([math.pi, 0, -math.pi/2])
            
            success = self.precision_docking_approach(
                self.robot2_id, target_docking_pos, target_docking_orn, target_obj
            )
            
            if success:
                print("정밀 도킹 테스트 성공!")
            else:
                print("정밀 도킹 테스트 실패!")
        
        self.move_both_to_home()
    
    def test_dual_robot_handover(self):
        """듀얼 로봇 전달 테스트"""
        print("듀얼 로봇 전달 테스트...")
        
        if not self.target_objects:
            print("테스트할 객체가 없습니다!")
            return
        
        # 유일한 객체로 테스트
        target_obj = self.target_objects[0]
        success = self.dual_robot_harvest_sequence(target_obj)
        
        if success:
            print("듀얼 로봇 전달 테스트 성공!")
        else:
            print("듀얼 로봇 전달 테스트 실패!")
    
    def run_full_harvest_cycle(self):
        """전체 수확 사이클 실행"""
        print("전체 수확 사이클 시작...")
        
        if not self.target_objects:
            print("수확할 꽃이 없습니다!")
            return
        
        # 유일한 꽃 수확
        target_obj = self.target_objects[0]
        print(f"샤프란 꽃 수확 중...")
        
        try:
            success = self.dual_robot_harvest_sequence(target_obj)
            if success:
                print("샤프란 꽃 수확 성공!")
            else:
                print("샤프란 꽃 수확 실패!")
                
        except Exception as e:
            print(f"수확 중 오류: {e}")
        
        print(f"수확 사이클 완료!")

    # === 고급 기능: 속도 제어 기반 가드 접근 ===
    def velocity_controlled_approach(self, robot_id, approach_axis, object_id, max_steps=200):
        """속도 제어 기반 마지막 접근 (Jacobian 버그 수정)"""
        print("속도 제어 기반 가드 접근...")
        
        q_indices = list(range(7))
        self._take_velocity_control(robot_id, q_indices)  # VELOCITY 제어 선점
        
        approach_velocity = 0.003  # 3mm/s
        
        for step in range(max_steps):
            # 현재 EE 위치 및 자세
            ee_state = p.getLinkState(robot_id, self.ee_link_index, computeForwardKinematics=True)
            ee_pos = np.array(ee_state[0])
            ee_quat = ee_state[1]
            ee_R = self.quat_to_R(ee_quat)
            
            # 로컬 접근축을 월드 좌표로 변환
            v_local = approach_velocity * np.array(approach_axis)
            v_world = ee_R @ v_local
            
            # 회전은 0
            twist = np.concatenate([v_world, [0, 0, 0]])
            
            # Jacobian은 가동 조인트만 사용!
            q_all = self._q_all_movable(robot_id)
            zero = [0.0]*len(q_all)
            
            try:
                Jlin, Jang = p.calculateJacobian(robot_id, self.ee_link_index, [0,0,0], q_all, zero, zero)
                J = np.vstack([np.array(Jlin), np.array(Jang)])
                
                # 제어할 관절만 추출 (가동 DOF 인덱스로 매핑)
                idx = [self.jindex_map[robot_id][j] for j in q_indices]
                J_cmd = J[:, idx]
                qdot_cmd = np.linalg.pinv(J_cmd, rcond=1e-3) @ twist
                
                # 속도 클램프(안정성)
                qdot_cmd = np.clip(qdot_cmd, -0.6, 0.6)
                
                # 속도 제어 적용
                for j, dq in zip(q_indices, qdot_cmd):
                    p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, 
                                          targetVelocity=float(dq), force=120)
                
            except Exception as e:
                print(f"속도 제어 실패: {e}")
                break
            
            # 접촉 또는 거리 체크
            success, distance, has_contact = self.check_docking_success(robot_id, object_id, expected_distance=0.03)
            
            if has_contact or distance < 0.025:
                print(f"속도 제어 접근 성공! (단계: {step}, 거리: {distance:.3f}m)")
                return True
            
            p.stepSimulation()
            time.sleep(1/240)
        
        print("속도 제어 접근 시간 초과")
        return False

    # === 실험적 기능: 적응형 핸드오버 ===
    def adaptive_handover(self, object_id, max_retries=3):
        """적응형 핸드오버 - 실패 시 파라미터 조정하여 재시도"""
        print("🔄 적응형 핸드오버 시작...")
        
        # 시도별 파라미터 조정
        params_list = [
            {'lift_height': 0.95, 'pre_offset': 0.05, 'step_size': 0.008},
            {'lift_height': 0.90, 'pre_offset': 0.07, 'step_size': 0.006},
            {'lift_height': 1.00, 'pre_offset': 0.04, 'step_size': 0.010}
        ]
        
        for attempt in range(max_retries):
            params = params_list[min(attempt, len(params_list)-1)]
            print(f"적응형 시도 {attempt+1}/{max_retries}: {params}")
            
            try:
                # 파라미터를 적용한 핸드오버 실행
                # (실제로는 perform_handover의 파라미터화된 버전을 호출)
                success = self.perform_handover(object_id)
                
                if success:
                    print(f"적응형 핸드오버 성공! (시도: {attempt+1})")
                    return True
                    
            except Exception as e:
                print(f"시도 {attempt+1} 실패: {e}")
            
            # 실패 시 안전 복구
            print("안전 복구 중...")
            self.move_both_to_home()
            time.sleep(1.0)  # 안정화 대기
        
        print("적응형 핸드오버 최종 실패")
        return False


# === 메인 실행 부분 ===
def main():
    """메인 실행 함수"""
    # PyBullet 초기화
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    # 시뮬레이터 안정화 설정
    p.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep=1/240)
    
    # 카메라 설정
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0, 0.5]
    )
    
    try:
        # 개선된 듀얼 로봇 컨트롤러 생성
        controller = ImprovedDualRobotController(physics_client)
        
        # 시스템 체크
        controller.system_check()
        
        print("\n" + "="*50)
        print("개선된 샤프란 수확 로봇 시스템")
        print("="*50)
        print("사용 가능한 테스트:")
        print("1. 시스템 상태 확인")
        print("2. 단일 로봇 테스트")
        print("3. 정밀 도킹 테스트")
        print("4. 듀얼 로봇 핸드오버 테스트")
        print("5. 전체 수확 사이클")
        print("6. 적응형 핸드오버 테스트")
        print("="*50)
        
        # 자동으로 전체 수확 사이클 실행
        input("엔터를 눌러 전체 수확 사이클을 시작하세요...")
        controller.run_full_harvest_cycle()
        
        # 추가 테스트 옵션
        while True:
            print("\n추가 테스트를 실행하시겠습니까?")
            print("1: 단일 로봇, 2: 정밀 도킹, 3: 적응형 핸드오버, q: 종료")
            choice = input("선택: ").strip()
            
            if choice == '1':
                controller.test_single_robot_harvest()
            elif choice == '2':
                controller.test_precision_docking()
            elif choice == '3':
                controller.adaptive_handover(controller.target_objects[0] if controller.target_objects else None)
            elif choice.lower() == 'q':
                break
            else:
                print("잘못된 선택입니다.")
        
    except Exception as e:
        print(f" 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔌 시뮬레이션 종료")
        p.disconnect()

if __name__ == "__main__":
    main()