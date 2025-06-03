from workflows.experiment_workflow import ExperimentCodingWorkflow
from agno.run.response import RunEvent
from agno.utils.log import logger, log_info
from agno.utils.pprint import pprint_run_response
from workflows.main_workflow import AgentLab
from config import OUTPUT_DIR
import json

if __name__ == "__main__":
    # 创建 Workflow 实例，传入 research_topic 并启用 debug_mode 以查看详细日志
    lab = AgentLab(
        research_topic="Novel Ensemble Methods of Decision Trees",
        lab_dir=OUTPUT_DIR,
        dataset_dir=[r"C:\data\adult.csv"],
        debug_mode=True,
    )

    print("Running workflow...")
    try:

        final_response = lab.run(user_id="test_user")

        # 打印最终结果
        if final_response:
            print("\nWorkflow finished.")
            print("Final Workflow Response Content:")
            # 假设 RunResponse 有方法或属性获取主要文本内容
            # 使用 get_content_as_string 方便打印
            print(
                final_response.get_content_as_string()
                if hasattr(final_response, "get_content_as_string")
                else final_response.content
            )
            # 你也可以打印其他详情，如工具调用、metrics 等
            if hasattr(final_response, "metrics") and final_response.metrics:
                print("\nMetrics:", final_response.metrics)

        print("\nFinal session state after run:")
        print(f"\nResearch Topic: {lab.session_state.get('research_topic')}")
        print(
            f"\nLit Review Papers Collected: {lab.session_state.get('lit_review', [])}"
        )
        print(f"\nLit Review Summary: {lab.session_state.get('lit_review_sum')}")
        print(f"\nPlan: {lab.session_state.get('plan')}")
        print(
            f"\ndata_preparation_result: {lab.session_state.get('data_preparation_result')}"
        )
        print(f"\nexperiment code: {lab.session_state.get('final_experiment_code','')}")
        print(
            f"\nfinal_experiment_execution_result: {lab.session_state.get('final_experiment_execution_result', {})}"
        )
        print(
            f"\ninterpretation content: {lab.session_state.get('final_interpretation_content', '')}"
        )
        print(f"\nPhase Status: {lab.session_state.get('phase_status')}")

    except Exception as e:
        print(f"\nAn unhandled exception occurred during workflow execution: {e}")
        # 使用 logger 记录详细异常信息
        logger.exception(e)
