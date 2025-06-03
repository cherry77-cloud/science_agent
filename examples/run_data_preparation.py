import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from agno.utils.pprint import pprint_run_response
from agno.utils.log import logger
from agno.cli.console import console  # For cleaner output
from agno.run.response import RunEvent
from workflows.data_preparation_workflow import DataAnalysisWorkflow


def main():
    # 只使用场景2: sklearn toy dataset
    plan_for_toy_dataset = """
    分析一个小型的机器学习数据集，探索数据特征和模式。
    使用sklearn的内置数据集进行基础的数据分析和可视化。
    包括数据探索、特征分析、缺失值检查等基本步骤。
    """
    topic_for_toy_dataset = "基础数据分析与探索"

    # 创建 Workflow 实例
    workflow = DataAnalysisWorkflow(
        name="基础数据分析流程",
        description="使用sklearn数据集进行基础数据分析",
        max_steps_per_dataset=5,  # 减少步数
        debug_mode=True,  # 启用调试模式
    )
    print(f"Workflow '{workflow.name}' 已初始化")

    # 使用场景2的配置
    current_plan = plan_for_toy_dataset
    current_topic = topic_for_toy_dataset
    current_dataset_paths = None  # 不提供数据，让系统使用toy dataset

    print(f"\n--- 运行工作流: {current_topic} ---")
    print("将使用sklearn内置数据集")
    print(f"分析计划: {current_plan}")

    # 运行 Workflow
    try:
        results_iterator = workflow.run_workflow(
            plan=current_plan, topic=current_topic, dataset_paths=current_dataset_paths
        )

        final_workflow_output = None
        for chunk_response in results_iterator:
            if (
                isinstance(chunk_response.content, dict)
                and "status" in chunk_response.content
            ):
                logger.info(
                    f"工作流更新 ({chunk_response.content['status']}):"
                    f" {chunk_response.content.get('message', '')}"
                )
            elif (
                hasattr(chunk_response, "event")
                and chunk_response.event == RunEvent.workflow_completed.value
            ):
                final_workflow_output = chunk_response
                logger.info("\n--- 工作流完成 ---")
                break
            else:
                logger.info(f"中间输出: {str(chunk_response.content)[:200]}...")

        # 打印最终结果
        if final_workflow_output and isinstance(final_workflow_output.content, list):
            logger.info("\n--- 最终报告 ---")
            for i, report in enumerate(final_workflow_output.content):
                logger.info(f"\n--- 数据集 {i + 1} 报告 ---")
                if "error" in report:
                    logger.error(f"错误: {report['error']}")
                else:
                    logger.info("✅ 分析完成")
                    if report.get("comprehensive_analysis_summary"):
                        logger.info(
                            f"分析摘要: {report.get('comprehensive_analysis_summary')[:200]}..."
                        )
        else:
            logger.error("工作流未产生预期的报告")

    except Exception as e:
        logger.error(f"运行工作流时发生错误: {e}")


if __name__ == "__main__":
    main()
