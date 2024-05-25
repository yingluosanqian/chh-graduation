import matplotlib.pyplot as plt


def draw_line_chart(data, save_path):
    # 创建一个新的图形和子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制折线图
    ax.plot(range(1, len(data) + 1), data, marker='o', linestyle='-')

    # 添加标题和轴标签
    ax.set_title('Line Plot of Data')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # 显示图形
    plt.grid(True)
    plt.savefig(save_path)  # 将生成的图片保存为plot_image.png文件
    plt.show()


def draw_bar_chart(data, save_path):
    # 创建一个新的图形和子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 生成柱状图
    plt.bar(range(len(data)), data)

    # 添加标题和轴标签
    plt.title('Bar Chart of Data')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示图形
    plt.grid(True)
    plt.savefig(save_path)  # 将生成的图片保存为plot_image.png文件
    plt.show()
