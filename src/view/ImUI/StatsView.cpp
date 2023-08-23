//
// Created by hepiteau on 23/08/23.
//

#include "StatsView.h"

#include <implot/implot.h>
#include <memory>
#include <utility>

StatsView::StatsView(std::shared_ptr<Statistics> statistics) : m_stats(std::move(statistics)) {

}

void StatsView::Render() {

    static float history = 100.0f;
    static ImPlotAxisFlags flags = ImPlotAxisFlags_None | ImPlotAxisFlags_AutoFit;

    auto sdata = m_stats->GetPSNRBuffer();
    auto t = m_stats->GetTime();

    auto tt = m_stats->GetTTime();
    ScrollingBuffer* loadBatchB = m_stats->GetLoadBatchBuffer();
    ScrollingBuffer* uploadDescB = m_stats->GetUploadDescBuffer( );
    ScrollingBuffer* zeroGradB = m_stats->GetZeroGradientBuffer( );
    ScrollingBuffer* forwardB = m_stats->GetForwardBuffer( );
    ScrollingBuffer* rayBackwardB = m_stats->GetRayBackwardBuffer( );
    ScrollingBuffer* volBackwardB = m_stats->GetVolBackwardBuffer( );
    ScrollingBuffer* AdamUpdateB = m_stats->GetAdamUpdateBuffer( );

    if (ImPlot::BeginPlot("PSNR ##Scrolling", ImVec2(-1,200))) {
        ImPlot::SetupAxes(NULL, NULL, flags, flags);
        ImPlot::SetupAxisLimits(ImAxis_X1,t - history, t, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1,0,40);
        ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);
        ImPlot::PlotShaded("##PSNRS", &sdata->Data[0].x, &sdata->Data[0].y, sdata->Data.size(), -INFINITY, 0, sdata->Offset, 2 * sizeof(float));
        ImPlot::PlotLine("PSNR", &sdata->Data[0].x, &sdata->Data[0].y, sdata->Data.size(), 0, sdata->Offset, 2*sizeof(float));
        ImPlot::EndPlot();
    }


    static ImPlotShadedFlags flags2 = 0;

    if (ImPlot::BeginPlot("Time")) {
        ImPlot::SetupAxes("Step","Time", flags, flags);
        ImPlot::SetupAxisLimits(ImAxis_X1,t - history, t, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1,0,200);
        ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);

        ImPlot::PlotLine("Load Batch Time", &loadBatchB->Data[0].x, &loadBatchB->Data[0].y, loadBatchB->Data.size(), 0, loadBatchB->Offset, 2 * sizeof (float));
        ImPlot::PlotLine("Upload Descriptors Time", &uploadDescB->Data[0].x, &uploadDescB->Data[0].y, uploadDescB->Data.size(), 0, uploadDescB->Offset, 2 * sizeof (float));
        ImPlot::PlotLine("Zero Gradients Time", &zeroGradB->Data[0].x, &zeroGradB->Data[0].y, zeroGradB->Data.size(), 0, zeroGradB->Offset, 2 * sizeof (float));
        ImPlot::PlotLine("Forward Time", &forwardB->Data[0].x, &forwardB->Data[0].y, forwardB->Data.size(), 0, forwardB->Offset, 2 * sizeof (float));
        ImPlot::PlotLine("Ray Backward Time", &rayBackwardB->Data[0].x, &rayBackwardB->Data[0].y, rayBackwardB->Data.size(), 0, rayBackwardB->Offset, 2 * sizeof (float));
        ImPlot::PlotLine("Volume Backward Time", &volBackwardB->Data[0].x, &volBackwardB->Data[0].y, volBackwardB->Data.size(), 0, volBackwardB->Offset, 2 * sizeof (float));
        ImPlot::PlotLine("Adam Update Time", &AdamUpdateB->Data[0].x, &AdamUpdateB->Data[0].y, AdamUpdateB->Data.size(), 0, AdamUpdateB->Offset, 2 * sizeof (float));

        ImPlot::EndPlot();
    }

}
