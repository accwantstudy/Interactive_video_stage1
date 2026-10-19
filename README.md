# Interactive Video — Stage 1

本项目旨在验证基于 **Interactive Motion Generation + Pose-Conditioned Video Rendering** 实现可交互视频生成的可行性。

现阶段采用两个独立模块构建完整的生成流程：

1. **ARDY Motion Prior**：基于 ARDY 模型生成可交互、可控制的人体 Motion，支持原有的交互控制方式以及新增的 UI Dragging 等功能，并将生成的 Motion 转换为 DW-Pose 表示。
2. **Wan Fun Video Renderer**：使用 Wan Fun Control 模型，以 ARDY 生成的 DW-Pose 序列作为条件进行视频生成，实现从交互式 Motion 到 RGB Video 的转换。

### Pipeline

```text
User Interaction
       │
       ▼
Interactive ARDY Motion Prior
       │
       ▼
Motion-to-DW-Pose Conversion
       │
       ▼
Wan Fun Video Renderer
       │
       ▼
Generated Video
```

### Repository Structure

```text
interactive_video_stage1/
│
├── ardy_m/                 # Interactive motion generation
│   └── README.md
│
├── video_render/       # Pose-conditioned video rendering
│   └── README.md
│
└── README.md               # Overall pipeline and project introduction
```

两个模块分别基于原始 ARDY 和 Wan Fun 项目进行修改和扩展，各自保留独立的代码结构。

当前阶段主要验证交互式 Motion Control 与 Pose-Conditioned Video Generation 结合的可行性，为后续端到端、实时交互式视频生成模型的开发提供基础。
