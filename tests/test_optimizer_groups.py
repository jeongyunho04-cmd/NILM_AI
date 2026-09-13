# -*- coding: utf-8 -*-
"""옵티마이저 조립 회귀 시험 (13.99).

**왜 있나.** `run_train_seq` 가 파라미터 무리를 이렇게 지었다:

    groups = [{"params": heads.parameters(), ...}]   # <- 제너레이터
    for g in groups:                                  # 중복검사
        for q in g["params"]: ...                     # <- 여기서 소진된다
    opt = AdamW(groups)                               # <- 빈 무리를 받는다

예외도 경고도 안 난다. `ChainHeads` 가 **12에포크 내내 초기값 그대로**였고
`seq_fix_*` 여섯 판이 전부 그렇게 나왔는데 표는 멀쩡해 보였다 — 몸통은 배웠고
사슬은 `emit=0 · base_scale=1 · switch_bias=-4` 라 **게이트 + 고정 이력현상**으로
동작했기 때문이다. 조용한 실패라 두 절(13.96·13.98)을 그 위에 썼다.

여기서 거는 그물은 둘이다.
  [1] 제너레이터를 무리에 넣고 훑으면 AdamW 가 빈 무리를 받는다 — **버그의 재현**.
  [2] `run_train_seq` 의 조립 규약: 무리가 비면 안 되고, AdamW 가 쥔 텐서 수가
      넣은 수와 같아야 한다.
"""
import torch
import torch.nn as nn


def _assemble(materialize):
    """`run_train_seq` 396~425 의 조립을 그대로 흉내낸다."""
    heads, model = nn.Linear(4, 4), nn.Linear(4, 4)
    hp = heads.parameters()
    groups = [{"params": list(hp) if materialize else hp, "lr": 3e-4},
              {"params": [q for q in model.parameters()], "lr": 1e-4}]
    seen = set()
    for g in groups:                                  # 중복검사
        for q in g["params"]:
            assert id(q) not in seen
            seen.add(id(q))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    return heads, model, opt


def test_generator_param_group_is_silently_emptied():
    """[1] 버그의 재현 — 이 성질이 사라지면 아래 그물은 필요 없어진 것이다."""
    heads, _, opt = _assemble(materialize=False)
    assert len(opt.param_groups[0]["params"]) == 0, "제너레이터가 더는 소진되지 않는다"
    assert len(opt.param_groups[1]["params"]) == 2, "리스트 무리는 멀쩡해야 한다"
    # 그리고 **조용하다** — 한 스텝 돌려도 예외 없이 가중치만 안 움직인다.
    w0 = heads.weight.detach().clone()
    opt.zero_grad(); heads(torch.randn(2, 4)).sum().backward(); opt.step()
    assert torch.equal(heads.weight.detach(), w0), "빈 무리인데 움직였다"


def test_materialized_param_group_actually_trains():
    """[2] 고침 — 리스트로 굳히면 AdamW 가 쥐고, 한 스텝에 움직인다."""
    heads, model, opt = _assemble(materialize=True)
    got = sum(len(g["params"]) for g in opt.param_groups)
    assert got == 4, "AdamW 가 쥔 텐서 %d != 4" % got
    for g in opt.param_groups:
        assert g["params"], "빈 무리가 있다"
    w0 = heads.weight.detach().clone()
    opt.zero_grad()
    (heads(torch.randn(2, 4)).sum() + model(torch.randn(2, 4)).sum()).backward()
    opt.step()
    assert not torch.equal(heads.weight.detach(), w0), "머리가 안 움직인다"


def test_run_train_seq_materializes_every_group():
    """`run_train_seq` 소스가 무리를 리스트로 굳히는가 — 되돌아가면 여기서 걸린다."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "run_train_seq.py"
    t = src.read_text(encoding="utf-8")
    for bad in ('{"params": heads.parameters()',
                '{"params": bghead.parameters()'):
        assert bad not in t, "제너레이터를 그대로 무리에 넣는다: %s" % bad
    assert 'g["params"] = list(g["params"])' in t, "중복검사가 무리를 안 굳힌다"
    assert "파라미터 무리가 비었다" in t, "빈 무리 그물이 없다"


def test_chain_heads_move_under_crf_loss():
    """사슬 머리가 CRF 손실에서 **정말** 기울기를 받나 (기제 확인).

    조립과 별개다 — 조립이 맞아도 손실 쪽에서 그래프가 끊기면 같은 증상이 난다.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.model.chain import ChainHeads, crf_nll
    from src.model.transition import N_FEAT
    torch.manual_seed(0)
    B, T, K, Z = 2, 8, 3, 32
    h = ChainHeads(Z, N_FEAT + 2, K, hidden=16, score_norm=0)
    em, on, off, ini = h(torch.randn(B, T, Z), torch.randn(B, T, N_FEAT + 2),
                         torch.randn(B, T, K))
    crf_nll(em, on, off, torch.rand(B, T, K) > 0.5, ini).mean().backward()
    #: `emit_scale` 만은 초기에 0 이 정상이다 — `emit.weight` 가 0 이라 곱이 0 이다.
    dead = [n for n, p in h.named_parameters()
            if n != "emit_scale" and (p.grad is None or float(p.grad.abs().max()) == 0.0)]
    assert not dead, "CRF 손실이 머리에 안 닿는다: %s" % dead
