# encoding: utf8

import paddle
import numpy as np
import paddle.fluid as fluid
from matplotlib import pyplot as plt

MODEL_DIR = "fit_a_line.inference.model"

infer_results = []
groud_truths = []

# 绘制真实值和预测值对比图

use_cuda = False  # use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

infer_exe = fluid.Executor(place)  # 创建推测用的executor
inference_scope = fluid.core.Scope()  # Scope指定作用域


def draw_infer_result(groud_truths, infer_results):
    title = 'Boston'
    plt.title(title, fontsize=24)
    x = np.arange(1, 20)
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results, color='green', label='training cost')
    plt.grid()
    plt.show()


with fluid.scope_guard(inference_scope):  # 修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。
    # 从指定目录中加载推理model(inference model)
    [inference_program,  # 推理的program
     feed_target_names,  # 需要在推理program中提供数据的变量名称
     fetch_targets] = fluid.io.load_inference_model(  # fetch_targets: 推断结果
        MODEL_DIR,  # MODEL_DIR:模型训练路径
        infer_exe)  # infer_exe: 预测用executor
    # 获取预测数据
    infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),  # 获取uci_housing的测试数据
                                batch_size=200)  # 从测试数据中读取一个大小为200的batch数据
    # 从test_reader中分割x
    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype("float32")
    test_y = np.array([data[1] for data in test_data]).astype("float32")
    results = infer_exe.run(inference_program,  # 预测模型
                            feed={feed_target_names[0]: np.array(test_x)},  # 喂入要预测的x值
                            fetch_list=fetch_targets)  # 得到推测结果

    print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        # print("%d: %.2f" % (idx, val))
        infer_results.append(val)

    print("ground truth:")
    for idx, val in enumerate(test_y):
        # print("%d: %.2f" % (idx, val))
        groud_truths.append(val)

    for truth_predict in zip(groud_truths, infer_results):
        print(truth_predict[0][0], truth_predict[1][0])

    # draw_infer_result(groud_truths, infer_results)
