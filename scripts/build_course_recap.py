"""Build a local-only course recap from existing, hash-verified evidence."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / 'outputs/adaptive-reference-20261113/tst100k/training-coverage-audit-v2'
IDENTITY = ROOT / 'docs/research/2026-11-18-route-handoff-evidence.json'
BACKGROUND = ROOT / 'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
BACKGROUND_EVIDENCE = ROOT / 'docs/research/2026-11-16-background-intervention-v2-evidence.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label):
    return f'<a href="{html.escape(path.resolve().as_uri(), quote=True)}">{html.escape(label)}</a>'


def background_demo():
    report = BACKGROUND / 'results.json'
    evidence = json.loads(BACKGROUND_EVIDENCE.read_text(encoding='utf-8'))
    assert digest(report) == evidence['results_sha256']
    results = json.loads(report.read_text(encoding='utf-8'))
    assert results['strength'] == 0.8 and results['model_sha256'] == evidence['model_sha256']
    case = next(case for case in results['cases'] if case['pair'] == '872-1268')
    variants = {item['name']: item for item in case['variants']}
    original = variants['identity']
    changed = variants['background_magenta_100']
    assert changed['intervention']['person_max_change'] == 0
    assert not original['certified_aligned'] and not changed['certified_aligned']
    panels = [
        (original['reference'], '原参考图'),
        (changed['reference'], '同一参考图，仅背景往品红改'),
        (original['methods']['adaptive']['file'], '用原参考图的结果'),
        (changed['methods']['adaptive']['file'], '用改背景参考图的结果'),
    ]
    images = []
    for image, label in panels:
        path = BACKGROUND / image['path'].replace('\\', '/')
        assert digest(path) == image['sha256']
        uri = html.escape(path.resolve().as_uri(), quote=True)
        images.append(f'<div><p>{label}</p><a href="{uri}"><img src="{uri}" width="256" height="256" alt="{label}"></a></div>')
    regions = changed['methods']['adaptive']['regions']
    return ''.join(images), regions['neutral_proxy']['rgb_mae'], digest(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/course-recap-20261118')
    args = parser.parse_args()
    identity = json.loads(IDENTITY.read_text(encoding='utf-8'))
    rows = identity['panels']
    assert len(rows) == 64 and len({r['index'] for r in rows}) == 64
    for row in rows:
        assert digest(PANELS / f"{row['index']}.jpg") == row['panel_sha256']
    audit_path = PANELS / 'results.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    gate = audit['development_gate']
    assert audit['finished'] and not audit['app_promoted']
    assert not gate['proceed_to_independent_evaluation']
    assert not gate['original_C10_gate_passed']
    assert gate['improved'] == 50 and gate['severe_regressions'] == 2
    assert audit['cells']['expanded']['best_epoch'] == 4
    featured = [(22927, '旧报告首例'), (88391, '旧报告首例'), (3026, '旧报告严重反例'), (27824, '旧报告严重反例')]
    assert {n for n, _ in featured}.issubset({r['index'] for r in rows})
    figures = []
    for index, label in featured:
        uri = (PANELS / f'{index}.jpg').resolve().as_uri()
        figures.append(f'<article><h3>{index} · {label}</h3><figure><a href="{uri}"><img src="{uri}" width="1024" height="550" loading="lazy" alt="案例{index}：目标、参考、数据集目标、当前13号与三个研究候选的原始对照面板"></a><figcaption>{index}为{label}，沿用原报告选择，未按本页展示效果重新挑图。</figcaption></figure></article>')
    all_links = ' '.join(link(PANELS / f"{row['index']}.jpg", str(row['index'])) for row in rows)
    background_images, neutral_change, background_sha256 = background_demo()
    source = Template('''<!doctype html>
<html lang="zh-CN" data-register="editorial"><head><title>映色：已经做成什么，仍卡在哪里</title>
<link rel="stylesheet" href="kit.css" data-ve-kit>
<style>
body{font-family:"Microsoft YaHei","Noto Sans CJK SC",Georgia,serif}h1,h2{font-family:Georgia,"Microsoft YaHei",serif}
main{max-width:1120px;margin:auto;padding:var(--s-6)}header{max-width:850px}h1{font-size:var(--t-h1)}
section{margin-top:var(--s-8)}.stages{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--s-5);border-top:1px solid var(--border-bright);padding-top:var(--s-5)}
.stage h3{margin:var(--s-3) 0}.stage p{max-width:28ch}.stage strong{display:block;font-size:var(--t-h3)}
.facts{display:grid;grid-template-columns:1fr 1fr;gap:var(--s-6)}.note{padding:var(--s-4);background:var(--surface);border:1px solid var(--border-bright)}
details{border-top:1px solid var(--border-bright);padding:var(--s-4) 0}summary{cursor:pointer;font-weight:600}
img{display:block;width:100%;height:auto;margin-top:var(--s-4)}figcaption{max-width:85ch}.links{line-height:2.4}.links a{display:inline-block;margin-right:var(--s-3)}
.background-demo{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));max-width:680px;gap:var(--s-4)}
.background-demo p{margin:0}.background-demo img{margin-top:var(--s-2)}
a,code{overflow-wrap:anywhere}.sources p{max-width:90ch}@media(max-width:760px){.stages,.facts{grid-template-columns:1fr}main{padding:var(--s-4)}}
@media print{details{break-inside:avoid}section{margin-top:var(--s-6)}}
</style></head><body><main>
<header><p class="eyebrow">映色 · 课程汇报 · 2026-11-18</p><h1>空间调色已可运行。<br>跨场景偏色仍未解决。</h1>
<p class="lead">已有软件可以体验、微调和导出；后续研究候选没有通过完整画质要求，未替换当前模型。</p>
<p class="meta">本机材料 · 引用已消费开发案例 · 无新训练或画质评分 · 素材不随Git上传</p></header>
<figure><div class="stages">
<div class="stage"><span class="chip info">早期方案</span><h3>全局颜色映射</h3><strong>整图共用变换</strong><p>能迁移整体色调；背景物体色也可能带到皮肤和白墙。</p></div>
<div class="stage"><span class="chip ok">现有13号</span><h3>空间自适应调色</h3><strong>不同位置可不同</strong><p>已经接入工作台，保留手动调整、预览与图片视频导出。</p></div>
<div class="stage"><span class="chip warn">研究未达标</span><h3>主体／背景协调</h3><strong>仍缺可靠调色决策</strong><p>有分区与内容线索，尚不能稳定区分本色、光照和可迁移风格。</p></div>
</div><figcaption>项目从全局变换推进到可运行的空间变换，尚未完成可靠的跨场景对象协调。</figcaption></figure>
<section><h2>为什么“识别人物”仍不够</h2><figure><div class="facts"><div class="note"><h3>知道在哪里</h3><p>分割和局部对应可提供人物、衣服、背景的位置或关联线索。旧研究已经使用过这些信息。</p></div><div class="note"><h3>还要决定调成什么</h3><p>参考里的蓝色可能来自天空本色，也可能来自冷调风格；人物边界并不直接给出应迁移的颜色关系。</p></div></div><figcaption>对象线索是输入信息，不能单独证明自动调色已经自然协调。</figcaption></figure>
<p>已尝试扩大配对训练、人物背景修正、局部内容对应及光照先验。部分指标有收益，完整门槛仍有失败；没有以换名字或调参数重复旧路线。</p></section>
<section><h2>参考图只改背景，目标白球服也偏色</h2><figure><div class="background-demo">$background_images</div><figcaption>上排为两张参考，下排为对同一目标的两次调色。两次均用现有自适应模型、强度 80%。参考人物像素保持不变，下排右图的白球服仍出现品红偏色。</figcaption></figure>
<p>两张输出在目标近中性色区域平均相差 $neutral_change 个 RGB 码值。区域用目标原图的颜色范围筛选。这个案例展示背景颜色如何影响输出，来自已使用的受控开发素材。$background_report</p></section>
<section><h2>平均误差下降，仍不足以上线</h2><figure><p class="ve-label">平均RGB误差（码值，越低越好）</p><ve-bars unit="" max="14" label="64条已消费开发配对的平均RGB误差，当前13号12.41128，扩展候选10.01818，越低越好"><ve-bar value="12.411">当前13号</ve-bar><ve-bar value="10.018" hot>扩展候选</ve-bar></ve-bars><figcaption>扩展第4轮相对当前13号平均RGB误差下降$gain%，但只是已消费开发范围的研究结果。</figcaption></figure>
<figure><ve-bars unit="组" max="20" label="相对原C10，换参考响应改善18组，预定最低要求20组"><ve-bar value="18" hot>实际改善</ve-bar><ve-bar value="20">最低要求</ve-bar></ve-bars><figcaption>换参考响应改善组数未过原定要求，因此没有用平均收益代替完整资格。</figcaption></figure>
<p class="note">同一开发范围：$improved／64条改善，$severe条严重退化。未进入最终独立验收；不能称为新软件默认效果，也不能把此开发集重新叫作独立测试集。</p></section>
<section><h2>画面和反例一起展示</h2><p>上排：目标、参考、数据集目标GT、当前13号。下排：原C10、旧数据续训、扩展候选。后三者均为研究比较对象；GT也不是所有自然审美的唯一答案。</p>
$figures
<details><summary>全部64份已有面板</summary><p class="links">$all_links</p><p>保留全部索引，不按本页观感删图或重算胜率。点击面板可打开原文件。</p></details></section>
<section><h2>现在现场演示什么</h2><figure><div class="facts"><div><h3>可以操作</h3><p>参考图＋目标图片／视频，强度、基础／HSL／曲线、撤销重做、PNG／MP4导出。先用合成示例熟悉功能。</p></div><div><h3>必须说明</h3><p>空间效果不能装进普通CUBE；RGB8链不等于HDR／Log管理；4K离线导出不等于实时处理。新画质总目标未完成。</p></div></div><figcaption>课程演示展示已有能力与真实限制，不把失败候选包装成产品升级。</figcaption></figure>
<p>$guide · $story · $handoff</p><p>软件入口：项目“00-从这里开始”中的13号研究版与14号稳定源码版。13号需本机研究环境及固定权重，其他电脑先按源码指南运行稳定版。</p></section>
<section class="sources"><h2>证据与版本</h2><p>$study · $functional · $progress</p>
<p>当前13号：多参考第6轮，权重SHA256 <code>d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9</code>。</p>
<p>扩展研究候选：第4轮，权重SHA256 <code>$checkpoint</code>。本页只复用已有面板；源码身份、面板哈希和数据来源仍以链接报告为准。</p>
<p>本页用于本机核对与汇报准备。课程展示或分发自然素材前须确认权限；本页生成结果及媒体不上传Git。</p></section>
</main></body></html>''')
    page = source.substitute(gain=f"{gate['relative_improvement'] * 100:.2f}", improved=gate['improved'], severe=gate['severe_regressions'], figures='\n'.join(figures), all_links=all_links, checkpoint=identity['checkpoint_sha256'], guide=link(ROOT/'00-从这里开始/0-体验与课程交付.md','体验与演示步骤'), story=link(ROOT/'docs/research/2026-11-17-project-story.md','可口述的项目主线'), handoff=link(ROOT/'docs/research/2026-11-18-route-handoff.md','完整失败交接'), study=link(ROOT/'docs/research/2026-11-14-training-coverage-study.md','扩展训练原报告'), functional=link(ROOT/'docs/research/2026-11-18-course-handoff.md','功能与交付核查'), progress=link(ROOT/'docs/RESEARCH_PROGRESS.md','当前进度'), background_images=background_images, neutral_change=f'{neutral_change:.2f}', background_report=link(ROOT/'docs/research/2026-11-16-background-intervention-v2.md', '受控诊断记录'))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / 'yingse-progress.src.html'
    path.write_text(page, encoding='utf-8')
    report = dict(scope='local recap only; no new inference, scoring or acceptance', panels_verified=len(rows), featured=[n for n, _ in featured], background_case='872-1268', background_results_sha256=background_sha256, background_images_verified=4, identity_sha256=digest(IDENTITY), audit_sha256=digest(audit_path), source_sha256=digest(path), generator_sha256=digest(Path(__file__)), app_promoted=False)
    (args.out_dir/'build-evidence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
