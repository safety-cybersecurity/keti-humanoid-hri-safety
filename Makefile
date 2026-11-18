.PHONY: help venv check test fixture validate lock lock-check clean

# 로컬 .venv 가 있으면 활성화 없이도 자동으로 쓴다.
# 없으면 시스템 python3 로 떨어진다 — CI 가 도는 경로가 이쪽이다.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# 생성 환경 설치 경로. 기기마다 다르면 환경변수나 make 인자로 덮어쓴다.
#   make lock ISAACLAB_PATH=/opt/IsaacLab
ISAACSIM_PATH     ?= $(HOME)/isaacsim
ISAACLAB_PATH     ?= $(HOME)/IsaacLab
ISAAC_ASSETS_PATH ?= $(HOME)/isaacsim_assets
LOCK_ARGS := --isaacsim-path $(ISAACSIM_PATH) --isaaclab-path $(ISAACLAB_PATH) \
             --assets-path $(ISAAC_ASSETS_PATH)

help:
	@echo "make venv        .venv 생성 + 의존성 설치 — 클론 후 1회"
	@echo "make check       검증기 자체 시험 + 단위 시험"
	@echo "make test        단위 시험과 정본 정합 시험"
	@echo "make fixture     합성 픽스처 생성 (.tmp/fixtures)"
	@echo "make validate    픽스처로 검증기 동작 확인"
	@echo "make lock        생성 환경 버전 수집 → environment.lock"
	@echo "make lock-check  현재 환경이 lock 과 같은지 대조 (대량 생성 전 필수)"
	@echo "make clean       생성물 삭제"

venv:
	@python3 -m venv .venv
	@.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt
	@echo "가상환경 준비 완료 — make 명령이 .venv 를 자동으로 사용합니다 (활성화 불필요)."

check:
	@$(MAKE) --no-print-directory validate
	@$(MAKE) --no-print-directory test

# 생성 파이프라인의 순수함수 단위 시험과 정본 정합 시험. Isaac 을 부팅하지 않으므로
# CI 에서 돈다.
# PYTHONPATH= blocks an inherited PYTHONPATH (e.g. a sourced ROS setup script)
# from pulling in unrelated pytest11 plugins that fail to import and break this.
test:
	@PYTHONPATH= $(PY) -m pytest tests -q

fixture:
	@$(PY) validator/make_fixture.py --with-media .tmp/fixtures

# 결함 픽스처 목록. `make validate BROKEN=clean` 으로 실패 경로를 시험할 수 있다.
BROKEN := broken_dof broken_region broken_label broken_seed broken_notice broken_time \
          broken_values broken_rules broken_contact_type broken_inputs_missing \
          broken_rule_inputs broken_scenario_rules broken_required_fields \
          broken_threshold_nonfinite broken_threshold_negative broken_missing_parameter broken_duplicate_rule_id \
          broken_empty_rules broken_zmp_half_nan broken_com_nan broken_safety_schema \
          broken_not_applicable_on_permitted broken_not_applicable_with_value \
          broken_contact_permitted_dropped \
          broken_episode_id_format broken_episode_id_seed \
          broken_run_start_missing broken_run_start_past_seq \
          broken_generator_dirty broken_generator_commit broken_step_null \
          broken_v_rel_nan broken_v_rel_negative broken_v_rel_length \
          broken_dof_mismatch broken_required_null broken_late_shape \
          broken_action_nan broken_canon_shapes broken_fps_zero \
          broken_late_joint_width broken_canon_shape_list broken_canon_json_fields \
          broken_canon_version broken_canon_list_as_object \
          broken_orphan_frame_episode broken_fps_text \
          broken_duplicate_episode_key broken_bad_triggered_rules \
          broken_termination_list broken_termination_member \
          broken_notes_member_dropped broken_notes_member_type \
          broken_canon_notes_members broken_redistributed_type \
          broken_canon_unrouted_shape broken_notes_empty_ssm \
          broken_notes_ssm_diverges_from_canon \
          broken_notes_bad_region broken_asset_missing \
          broken_notes_region_detail broken_notes_hold_contaminated \
          broken_notes_region_contradiction broken_canon_notes_bounds \
          broken_asset_canon_flag broken_notes_rule_id_type \
          broken_asset_license_one_episode broken_asset_flag_one_episode \
          broken_asset_unknown_path broken_notes_aggregate_without_regions \
          broken_canon_conditional_steps \
          broken_stop_table broken_directed_parameter_missing broken_canon_step_layers \
          broken_hri_column_missing broken_hri_violation_gate broken_hri_negative_approach \
          broken_hri_visible_fraction broken_hri_ssm_params broken_hri_task_step broken_hri_task_null \
          broken_hri_task_stages \
          broken_hri_contact_without_onset \
          broken_hri_spec_missing broken_hri_spec_shape broken_hri_grid_cell broken_hri_evidence_text \
          broken_layer_missing_column \
          broken_media_frame_count broken_media_pack_missing broken_media_bbox_out_of_range \
          broken_media_stray_pack \
          broken_asset_unreferenced broken_texture_missing broken_texture_bytes \
          broken_texture_unlisted broken_asset_left_out broken_asset_usd_bytes \
          broken_span_swapped broken_span_end_short broken_span_file_index \
          broken_span_rows_out_of_place \
          broken_total_frames broken_total_episodes \
          broken_task_count broken_task_table_absent broken_task_sentences_not_index \
          broken_task_index_outside broken_task_index_beyond_feature \
          broken_features_absent broken_feature_undeclared_column broken_feature_column_absent \
          broken_feature_shape broken_feature_dtype_name broken_feature_frame_index_width

# 결함 없는 픽스처. clean 은 카메라 없는 배포본(v0.1 꼴), clean_media 는 카메라 배포본
# (meta/cameras.json · videos/, 정본이 팩을 선언하지 않음), clean_media_pack 은 팩 하나를 선언한
# 카메라 배포본(packs/ · meta/packs.json 도), clean_hri 는 S1e 의 판정과 열을 싣는 hri 배포본,
# clean_hri_media 는 거기에 카메라 매체를 더한 배포본(카메라마다 영상 · 자세와 보이는 몫 열),
# clean_hri_lean 은 clean_hri 에서 시뮬레이터가 기록하는 선택 스텝 열 아홉을 뺀 배포본 —
# 여섯 다 전 검사에서 ERROR 0 이어야 한다.
CLEAN := clean clean_media clean_media_pack clean_hri clean_hri_media clean_hri_lean

validate: fixture
	@for d in $(CLEAN); do \
		echo "── $$d (ERROR 0 이어야 함) ──"; \
		$(PY) validator/validate_dataset.py .tmp/fixtures/$$d > .tmp/$$d.log 2>&1 \
			|| { tail -20 .tmp/$$d.log; echo "$$d 픽스처에서 ERROR — 검증기 회귀"; exit 1; }; \
		tail -2 .tmp/$$d.log; \
	done
	@echo "── 결함 픽스처 (전부 검출되어야 함) ──"
	@# 픽스처마다 출력을 .tmp/validate/<픽스처>.log 에 남긴다. 검증기는 exit 3 에 트레이스백을,
	@# 시그널에 faulthandler 스택을 stderr 로 찍는데(2026-09-07), 이 루프가 그것을 /dev/null 로
	@# 버리고 있어 네 번째 간헐 크래시(2026-09-14, broken_rule_inputs)도 스택 없이 지나갔다.
	@mkdir -p .tmp/validate; fail=0; \
	for d in $(BROKEN); do \
		printf "%-16s " $$d; \
		$(PY) validator/validate_dataset.py .tmp/fixtures/$$d > .tmp/validate/$$d.log 2>&1; \
		rc=$$?; \
		if [ $$rc -eq 1 ]; then \
			echo "검출"; \
		elif [ $$rc -eq 0 ]; then \
			echo "미검출 (문제)"; fail=1; \
		else \
			echo "exit $$rc — 검출이 아니라 검증기가 죽었습니다 (문제). 스택: .tmp/validate/$$d.log"; \
			tail -40 .tmp/validate/$$d.log; fail=1; \
		fi; \
	done; \
	exit $$fail

# environment.lock 은 Isaac Sim 이 설치된 기기에서만 갱신한다. CI 는 Isaac 이 없으므로
# make check 에 넣지 않는다 — 대량 생성 직전에 lock-check 를 수동으로 통과시킨다.
lock:
	@$(PY) generation/capture_environment.py $(LOCK_ARGS)

lock-check:
	@$(PY) generation/capture_environment.py $(LOCK_ARGS) --check

clean:
	@rm -rf .tmp docs/build
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "정리 완료"
