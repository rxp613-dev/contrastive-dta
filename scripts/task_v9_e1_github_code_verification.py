"""
V9 Task E1: GitHub Code Availability Verification for CL-DTA Methods
Created: 2026-10-21
Objective: Verify availability and completeness of CSCo-DTA and GraphCL-DTA code repositories
Method: Direct GitHub repository inspection via WebFetch
"""

import json
from datetime import datetime

# Verification results
verification_results = {
    "verification_date": "2026-10-21",
    "task": "E1: B2事实核验 - GitHub代码可用性验证",
    
    "CSCo-DTA": {
        "repository": "github.com/23AIBox/23AIBox-CSCo-DTA",
        "status": "公开可访问",
        "paper": "Wang et al., Briefings in Bioinformatics, 2024",
        "author_affiliation": "西北工业大学23AIBox实验室 (与论文作者Jingru Wang一致)",
        
        "contents": {
            "models.py": "模型定义 (包含跨尺度对比学习架构)",
            "inference.py": "推理脚本",
            "utils.py": "工具函数",
            "data_process.py": "数据处理脚本",
            "data.zip": "数据文件 (Davis/KIBA)",
            "README": "项目说明"
        },
        
        "completeness_assessment": {
            "has_training_script": False,
            "has_pretrained_weights": False,
            "has_model_definition": True,
            "has_inference_code": True,
            "has_data_files": True,
            "has_data_processing": True
        },
        
        "metadata": {
            "stars": 14,
            "commits": 21,
            "size_mb": 12.6,
            "last_update": "2024-04 (推测)"
        },
        
        "usability_for_reproduction": {
            "full_reproduction": "不可行 - 缺少训练脚本和预训练权重",
            "inference_only": "可行 - 可使用inference.py进行推理",
            "model_architecture_reference": "可行 - models.py可作为架构参考",
            "component_integration": "可行 - 可参考models.py实现跨尺度对比损失"
        },
        
        "conclusion": "代码部分公开，可用于架构参考和组件级集成，但无法完整复现训练过程"
    },
    
    "GraphCL-DTA": {
        "repository": "github.com/NovaRick/GraphCL-DTA",
        "status": "公开可访问",
        "paper": "Yang et al., IEEE JBHI, 2024 (preprint)",
        "author_affiliation": "GitHub用户NovaRick (身份未确认，需邮件核实是否为论文作者)",
        
        "contents": {
            "GraphCL-DTA.zip": "完整代码压缩包 (内容未公开，需解压验证)"
        },
        
        "completeness_assessment": {
            "has_training_script": "未知 (压缩包内)",
            "has_pretrained_weights": "未知 (压缩包内)",
            "has_model_definition": "未知 (压缩包内)",
            "has_inference_code": "未知 (压缩包内)",
            "has_data_files": "未知 (压缩包内)",
            "has_data_processing": "未知 (压缩包内)"
        },
        
        "metadata": {
            "stars": 1,
            "commits": 3,
            "size_kb": 937,
            "last_update": "未知"
        },
        
        "usability_for_reproduction": {
            "full_reproduction": "需验证 - 需下载并解压检查内容",
            "author_verification": "需邮件确认是否为论文作者官方仓库",
            "component_integration": "待验证 - 取决于压缩包内容完整性"
        },
        
        "conclusion": "仓库存在但作者身份未确认，内容完整性需下载验证"
    },
    
    "overall_assessment": {
        "previous_claim": "均无公开代码",
        "corrected_claim": "两个方法的代码均在GitHub上公开可获取，但完整性和可用性存在差异",
        
        "key_findings": [
            "CSCo-DTA: 官方代码存在，含模型定义和推理脚本，但缺训练脚本和预训练权重",
            "GraphCL-DTA: 代码仓库存在，但作者身份待确认，内容完整性需验证",
            "两个仓库均可用于组件级集成（参考架构和损失函数实现）",
            "完整训练复现仍需工程投入（实现训练循环、超参数调优等）"
        ],
        
        "implications_for_paper": {
            "limitations_statement": "必须修正：删除'均无公开代码'的错误表述",
            "correct_statement": "代码获取渠道存在，但CSCo-DTA缺训练脚本、GraphCL-DTA仓库身份待确认，完整训练复现仍需工程投入",
            "recommended_approach": "组件级集成（E8/E9配置）是性价比最高的方案，可在受控框架内公平比较"
        }
    },
    
    "next_steps": [
        "E1.4: 更新Limitations，修正事实错误",
        "E4: 基于CSCo-DTA models.py实现E8配置（跨尺度对比损失）",
        "E5: 下载GraphCL-DTA.zip验证内容，如完整则实现E9配置（分子语义对比损失）",
        "考虑联系原作者索取训练代码/权重（邮件留痕）"
    ]
}

# Save verification report
output_path = "/mnt/data/dta2/results/analysis/task_v9_e1_github_code_verification.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(verification_results, f, indent=2, ensure_ascii=False)

print("=" * 80)
print("V9 Task E1: GitHub Code Availability Verification - COMPLETED")
print("=" * 80)
print(f"\nReport saved to: {output_path}\n")

print("Key Findings:")
print("-" * 80)
print("1. CSCo-DTA (23AIBox/23AIBox-CSCo-DTA):")
print("   - 公开可访问，14 stars，12.6MB")
print("   - 包含: models.py, inference.py, data.zip, data_process.py")
print("   - 缺少: 训练脚本, 预训练权重")
print("   - 可用于: 架构参考, 组件级集成")
print()
print("2. GraphCL-DTA (NovaRick/GraphCL-DTA):")
print("   - 公开可访问，1 star，937KB")
print("   - 内容: GraphCL-DTA.zip (需解压验证)")
print("   - 作者身份: 待确认")
print("   - 可用于: 待验证")
print()
print("Conclusion:")
print("-" * 80)
print("❌ 原声明'均无公开代码' - 错误")
print("✅ 修正声明: 代码获取渠道存在，但完整性和可用性存在差异")
print()
print("Recommended Action:")
print("-" * 80)
print("组件级集成（E8/E9配置）是性价比最高的方案")
print("- 成本: 40-80 GPU小时 (vs 全量复现 100-200 GPU小时)")
print("- 优势: 在受控框架内公平比较，直接回应评审意见")
print("=" * 80)
