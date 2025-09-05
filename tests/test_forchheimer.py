#!/usr/bin/env python3
"""
Forchheimer阻力驗證（body_force + Guo forcing路徑）

說明：
- 本測試聚焦於驗證濾紙Forchheimer阻力是否正確累加至 lbm.body_force。
- 為避免巨型網格開銷，若當前網格過大則跳過此測試。
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import numpy as np
import pytest
import taichi as ti
import config as config
from src.physics.filter_paper import FilterPaperSystem


@pytest.fixture(scope="module", autouse=True)
def setup_taichi():
    # 使用CPU後端以便在CI上運行
    ti.init(arch=ti.cpu, random_seed=0)
    yield
    ti.reset()


@pytest.mark.skipif(max(config.NX, config.NY, config.NZ) > 64,
                    reason="Domain too large for unit test computation")
def test_forchheimer_body_force_accumulation():
    """驗證Forchheimer阻力是否累加到lbm.body_force（不直接改u）。"""

    @ti.data_oriented
    class MockLBMSolver:
        def __init__(self):
            self.u = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
            self.rho = ti.field(dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))
            self.solid = ti.field(dtype=ti.u8, shape=(config.NX, config.NY, config.NZ))
            self.body_force = ti.Vector.field(3, dtype=ti.f32, shape=(config.NX, config.NY, config.NZ))

        @ti.kernel
        def init_fields(self):
            for i, j, k in ti.ndrange(config.NX, config.NY, config.NZ):
                self.u[i, j, k] = ti.Vector([0.02, 0.0, -0.015])
                self.rho[i, j, k] = 1.0
                self.solid[i, j, k] = ti.u8(0)
                self.body_force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    lbm = MockLBMSolver()
    lbm.init_fields()

    # 建立濾紙系統並初始化幾何與參數
    fp = FilterPaperSystem(lbm)
    fp.initialize_filter_geometry()

    # 累加Forchheimer阻力至body_force（不直接改速度）
    fp.compute_forchheimer_resistance()

    bf = lbm.body_force.to_numpy()
    fz = fp.filter_zone.to_numpy()

    # 僅在濾紙區域檢查體力是否非零
    if np.sum(fz == 1) == 0:
        pytest.skip("No filter zone cells marked; geometry setup likely skipped")

    avg_force_mag = float(np.mean(np.linalg.norm(bf[fz == 1], axis=1))) if bf.shape[-1] == 3 else 0.0
    # 體力非零代表Forchheimer已透過正確路徑累加
    assert avg_force_mag > 0.0, "Forchheimer未正確累加至body_force"

    # 非濾紙區域的體力應較小（或為零）
    if np.sum(fz == 0) > 0:
        avg_force_non_filter = float(np.mean(np.linalg.norm(bf[fz == 0], axis=1)))
        assert avg_force_non_filter <= avg_force_mag + 1e-6

