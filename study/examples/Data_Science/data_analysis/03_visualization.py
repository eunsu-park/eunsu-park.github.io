"""
데이터 시각화 (Data Visualization)
Matplotlib and Seaborn Examples

데이터를 시각적으로 표현하는 방법을 다룹니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn은 선택적 (없으면 건너뜀)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not installed. Some examples will be skipped.")


# =============================================================================
# 1. 기본 선 그래프
# =============================================================================
def line_plot():
    """선 그래프"""
    print("\n[1] 선 그래프 (Line Plot)")
    print("=" * 50)

    # 데이터 준비
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 그래프 생성
    plt.figure(figsize=(10, 6))

    plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
    plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)

    plt.title('삼각 함수 그래프', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/01_line_plot.png', dpi=150)
    plt.close()
    print("저장: 01_line_plot.png")


# =============================================================================
# 2. 산점도
# =============================================================================
def scatter_plot():
    """산점도 (Scatter Plot)"""
    print("\n[2] 산점도 (Scatter Plot)")
    print("=" * 50)

    np.random.seed(42)

    # 데이터 생성
    n = 100
    x = np.random.randn(n)
    y = 2 * x + 1 + np.random.randn(n) * 0.5
    colors = np.random.rand(n)
    sizes = np.random.rand(n) * 200

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='색상 값')

    plt.title('산점도 예제', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/02_scatter_plot.png', dpi=150)
    plt.close()
    print("저장: 02_scatter_plot.png")


# =============================================================================
# 3. 막대 그래프
# =============================================================================
def bar_plot():
    """막대 그래프"""
    print("\n[3] 막대 그래프 (Bar Plot)")
    print("=" * 50)

    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [23, 45, 56, 78, 32]
    values2 = [17, 38, 42, 65, 28]

    x = np.arange(len(categories))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 그룹화된 막대 그래프
    ax = axes[0]
    ax.bar(x - width/2, values1, width, label='그룹1', color='steelblue')
    ax.bar(x + width/2, values2, width, label='그룹2', color='coral')
    ax.set_xlabel('카테고리')
    ax.set_ylabel('값')
    ax.set_title('그룹화된 막대 그래프')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # 누적 막대 그래프
    ax = axes[1]
    ax.bar(categories, values1, label='그룹1', color='steelblue')
    ax.bar(categories, values2, bottom=values1, label='그룹2', color='coral')
    ax.set_xlabel('카테고리')
    ax.set_ylabel('값')
    ax.set_title('누적 막대 그래프')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/03_bar_plot.png', dpi=150)
    plt.close()
    print("저장: 03_bar_plot.png")


# =============================================================================
# 4. 히스토그램
# =============================================================================
def histogram():
    """히스토그램"""
    print("\n[4] 히스토그램 (Histogram)")
    print("=" * 50)

    np.random.seed(42)

    # 정규 분포 데이터
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1.5, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 기본 히스토그램
    ax = axes[0]
    ax.hist(data1, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_xlabel('값')
    ax.set_ylabel('빈도')
    ax.set_title('기본 히스토그램')

    # 겹친 히스토그램
    ax = axes[1]
    ax.hist(data1, bins=30, alpha=0.5, label='μ=0, σ=1', color='blue')
    ax.hist(data2, bins=30, alpha=0.5, label='μ=2, σ=1.5', color='red')
    ax.set_xlabel('값')
    ax.set_ylabel('빈도')
    ax.set_title('두 분포 비교')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/04_histogram.png', dpi=150)
    plt.close()
    print("저장: 04_histogram.png")


# =============================================================================
# 5. 파이 차트
# =============================================================================
def pie_chart():
    """파이 차트"""
    print("\n[5] 파이 차트 (Pie Chart)")
    print("=" * 50)

    labels = ['Python', 'JavaScript', 'Java', 'C++', 'Other']
    sizes = [35, 25, 20, 10, 10]
    explode = (0.1, 0, 0, 0, 0)  # Python 강조
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 기본 파이 차트
    ax = axes[0]
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title('프로그래밍 언어 점유율')

    # 도넛 차트
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       wedgeprops=dict(width=0.5))
    ax.set_title('도넛 차트')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/05_pie_chart.png', dpi=150)
    plt.close()
    print("저장: 05_pie_chart.png")


# =============================================================================
# 6. 박스 플롯과 바이올린 플롯
# =============================================================================
def box_violin_plot():
    """박스 플롯과 바이올린 플롯"""
    print("\n[6] 박스 플롯과 바이올린 플롯")
    print("=" * 50)

    np.random.seed(42)

    # 데이터 생성
    data = [np.random.normal(0, std, 100) for std in range(1, 5)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 박스 플롯
    ax = axes[0]
    bp = ax.boxplot(data, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(['σ=1', 'σ=2', 'σ=3', 'σ=4'])
    ax.set_xlabel('그룹')
    ax.set_ylabel('값')
    ax.set_title('박스 플롯')

    # 바이올린 플롯
    ax = axes[1]
    vp = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['σ=1', 'σ=2', 'σ=3', 'σ=4'])
    ax.set_xlabel('그룹')
    ax.set_ylabel('값')
    ax.set_title('바이올린 플롯')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/06_box_violin.png', dpi=150)
    plt.close()
    print("저장: 06_box_violin.png")


# =============================================================================
# 7. 히트맵
# =============================================================================
def heatmap():
    """히트맵"""
    print("\n[7] 히트맵 (Heatmap)")
    print("=" * 50)

    np.random.seed(42)

    # 상관 행렬 데이터
    data = np.random.randn(5, 5)
    data = (data + data.T) / 2  # 대칭 행렬
    np.fill_diagonal(data, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matplotlib 히트맵
    ax = axes[0]
    im = ax.imshow(data, cmap='coolwarm', aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E'])
    plt.colorbar(im, ax=ax)
    ax.set_title('Matplotlib 히트맵')

    # 값 표시
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center')

    # Seaborn 히트맵 (있는 경우)
    ax = axes[1]
    if HAS_SEABORN:
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    xticklabels=['A', 'B', 'C', 'D', 'E'],
                    yticklabels=['A', 'B', 'C', 'D', 'E'])
        ax.set_title('Seaborn 히트맵')
    else:
        ax.text(0.5, 0.5, 'Seaborn not available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Seaborn 히트맵 (없음)')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/07_heatmap.png', dpi=150)
    plt.close()
    print("저장: 07_heatmap.png")


# =============================================================================
# 8. 서브플롯
# =============================================================================
def subplots_example():
    """서브플롯 레이아웃"""
    print("\n[8] 서브플롯 (Subplots)")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('다양한 그래프 모음', fontsize=16)

    np.random.seed(42)

    # (0, 0) 선 그래프
    ax = axes[0, 0]
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin')
    ax.plot(x, np.cos(x), label='cos')
    ax.set_title('선 그래프')
    ax.legend()

    # (0, 1) 산점도
    ax = axes[0, 1]
    ax.scatter(np.random.randn(50), np.random.randn(50))
    ax.set_title('산점도')

    # (0, 2) 막대 그래프
    ax = axes[0, 2]
    ax.bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
    ax.set_title('막대 그래프')

    # (1, 0) 히스토그램
    ax = axes[1, 0]
    ax.hist(np.random.randn(1000), bins=30)
    ax.set_title('히스토그램')

    # (1, 1) 파이 차트
    ax = axes[1, 1]
    ax.pie([30, 20, 25, 25], labels=['A', 'B', 'C', 'D'], autopct='%1.0f%%')
    ax.set_title('파이 차트')

    # (1, 2) 이미지
    ax = axes[1, 2]
    ax.imshow(np.random.rand(10, 10), cmap='viridis')
    ax.set_title('이미지')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/08_subplots.png', dpi=150)
    plt.close()
    print("저장: 08_subplots.png")


# =============================================================================
# 9. 3D 플롯
# =============================================================================
def plot_3d():
    """3D 플롯"""
    print("\n[9] 3D 플롯")
    print("=" * 50)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 5))

    # 3D 표면
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D 표면')

    # 3D 산점도
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    n = 100
    xs = np.random.randn(n)
    ys = np.random.randn(n)
    zs = np.random.randn(n)
    colors = np.random.rand(n)

    ax2.scatter(xs, ys, zs, c=colors, cmap='plasma')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D 산점도')

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Data_Analysis/examples/09_3d_plot.png', dpi=150)
    plt.close()
    print("저장: 09_3d_plot.png")


# =============================================================================
# 메인
# =============================================================================
def main():
    print("=" * 60)
    print("데이터 시각화 예제")
    print("=" * 60)

    # 한글 폰트 설정 (시스템에 따라 조정 필요)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    line_plot()
    scatter_plot()
    bar_plot()
    histogram()
    pie_chart()
    box_violin_plot()
    heatmap()
    subplots_example()
    plot_3d()

    print("\n" + "=" * 60)
    print("시각화 핵심 정리")
    print("=" * 60)
    print("""
    Matplotlib 기본:
    - plt.figure(): 새 그림 생성
    - plt.subplot(): 서브플롯 생성
    - plt.plot(), scatter(), bar(), hist(): 그래프 종류
    - plt.xlabel(), ylabel(), title(): 라벨과 제목
    - plt.legend(), grid(): 범례와 그리드
    - plt.savefig(), show(): 저장과 표시

    Seaborn 장점:
    - 더 예쁜 기본 스타일
    - 통계적 시각화 (regplot, kdeplot)
    - DataFrame과 쉬운 연동

    그래프 선택 가이드:
    - 추세: 선 그래프
    - 관계: 산점도
    - 비교: 막대 그래프
    - 분포: 히스토그램, 박스플롯
    - 비율: 파이 차트
    - 상관관계: 히트맵

    팁:
    - 적절한 색상 팔레트 선택
    - 라벨과 제목 명확하게
    - 범례 위치 최적화
    - DPI 설정으로 해상도 조절
    """)


if __name__ == "__main__":
    main()
