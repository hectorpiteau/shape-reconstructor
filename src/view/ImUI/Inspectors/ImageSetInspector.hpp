#pragma once

#include <string>
#include <vector>

#include "../../../include/imgui/imgui.h"
#include "../../../include/imgui/backends/imgui_impl_glfw.h"
#include "../../../include/imgui/backends/imgui_impl_opengl3.h"

#include "../../../include/icons/IconsFontAwesome6.h"

#include "../../../model/ImageSet.hpp"
#include "../../../model/Image.hpp"
#include "../../../model/Texture2D.hpp"

#include "../../../interactors/ImageSetInteractor.hpp"



class ImageSetInspector
{
public:
    ImageSetInspector(ImageSetInteractor *interactor) : m_interactor(interactor)
    {
        m_comboBoxImagesName = std::vector<char *>();
    };

    ImageSetInspector(const ImageSetInspector &) = delete;

    ~ImageSetInspector(){};

    void Update()
    {
        if (m_interactor->GetImageSet() == nullptr) return;

        /** Copy the full path to the current folder; */
        strcpy(m_folderPath, m_interactor->GetImageSet()->GetFolderPath().c_str());
        std::cout <<"ImageSet UPDATE: " << m_interactor->GetImageSet()->GetFolderPath() << std::endl;

        if (m_interactor->GetImageSet()->size() <= 0) return;

        
        if (m_comboBoxImagesName.size() > 0)
        {
            for (char *elem : m_comboBoxImagesName)
            {
                delete[] elem;
            }
            m_comboBoxImagesName.clear();
        }

        /** Retrieve images names for combobox. */
        for (size_t i = 0; i < m_interactor->GetImageSet()->size(); ++i)
        {
            const Image *img = (*m_interactor->GetImageSet())[i];

            char *tmp = new char[img->GetFilename().length()];
            strcpy(tmp, img->GetFilename().c_str());
            m_comboBoxImagesName.push_back(tmp);

            std::cout << img->GetFilename() << std::endl;
        }
    }

    void Render()
    {
        if (m_interactor == nullptr || m_interactor->GetImageSet() == nullptr)
        {
            ImGui::Text("Error: interactor is null.");
            return;
        }
        /** Check if the imageSet have been updated or not. */
        if(m_interactor->GetUpdatedImageSet()){
            Update();
            m_interactor->SetUpdatedImageSet(false);
        }

        ImGui::SeparatorText(ICON_FA_INFO " ImageSet - Informations");
        /** SceneObject id. */
        ImGui::TextUnformatted((std::string("Images count: ") + std::to_string(m_interactor->GetImageSet()->GetAmountOfImages())).c_str());

        /** Image tab with load status. */
        ImGui::Spacing();
        ImGui::SeparatorText(ICON_FA_FOLDER " Source");

        ImGui::InputText("Folder path", m_folderPath, 256); // IM_ARRAYSIZE(m_folderPath)

        if (ImGui::Button(ICON_FA_IMAGES " Load images"))
        {
            size_t count = m_interactor->LoadImages(m_folderPath);
            if (count > 0)
                m_loadStatus = std::string("loaded");
            else
                m_loadStatus = std::string("not loaded");

            Update();
        }

        ImGui::SameLine();
        ImGui::TextUnformatted(m_loadStatus.c_str());

        ImGui::Spacing();
        ImGui::SeparatorText(ICON_FA_IMAGE " Image Preview");
        if (m_interactor->GetImageSet() == nullptr)
        {
            ImGui::TextWrapped("No image selected.");
        }
        else
        {
            static ImGuiComboFlags flags = 0;

            /** Display image example. */
            if (m_comboBoxImagesName.size() > 0)
            {
                const char *combo_preview = m_comboBoxImagesName[m_comboBoxCurrent];

                if (ImGui::BeginCombo("Preview Image", combo_preview, flags))
                {
                    for (size_t i = 0; i < m_comboBoxImagesName.size(); ++i)
                    {
                        const bool is_selected = (m_comboBoxCurrent == i);

                        if (ImGui::Selectable(m_comboBoxImagesName[i], is_selected)){
                            if(m_comboBoxCurrent != i){
                                m_image = m_interactor->GetImageSet()->GetImage(i);
                                m_imageTex.LoadFromImage(m_image);
                            }
                            m_comboBoxCurrent = i;
                        }

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                if(m_image != nullptr){

                    ImGuiIO &io = ImGui::GetIO();

                    static float dsize = 0.4;
                    ImGui::SliderFloat("Displayed Size", &dsize, 0.001f, 2.0f);
                    ImGui::Checkbox("Hovered Zoom", &m_enableZoom);

                    float my_tex_w = m_imageTex.GetWidth() * dsize;
                    float my_tex_h = m_imageTex.GetHeight() * dsize;

                    ImVec2 pos = ImGui::GetCursorScreenPos();

                    ImGui::Image((void *)(intptr_t)m_imageTex.GetID(), ImVec2(my_tex_w, my_tex_h), ImVec2(0, 0), ImVec2(1, 1));

                    if (m_enableZoom && ImGui::IsItemHovered())
                    {
                        ImGui::BeginTooltip();
                        float region_sz = 64.0f;
                        float region_x = io.MousePos.x - pos.x - region_sz * 0.5f;
                        float region_y = io.MousePos.y - pos.y - region_sz * 0.5f;
                        float zoom = 4.0f;
                        if (region_x < 0.0f)
                        {
                            region_x = 0.0f;
                        }
                        else if (region_x > my_tex_w - region_sz)
                        {
                            region_x = my_tex_w - region_sz;
                        }
                        if (region_y < 0.0f)
                        {
                            region_y = 0.0f;
                        }
                        else if (region_y > my_tex_h - region_sz)
                        {
                            region_y = my_tex_h - region_sz;
                        }

                        ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
                        ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
                        ImVec2 uv0 = ImVec2((region_x) / my_tex_w, (region_y) / my_tex_h);
                        ImVec2 uv1 = ImVec2((region_x + region_sz) / my_tex_w, (region_y + region_sz) / my_tex_h);
                        ImGui::Image((void *)(intptr_t)m_imageTex.GetID(), ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1);
                        ImGui::EndTooltip();
                    }
                }
            }
            else
            {
                ImGui::Text("~ wait for loading ~ ");
            }

            /** Displayed image resolution. */
            // ImGui::Text((std::string("Image size: (") + std::to_string(m_image->width) + std::string(", ") + std::to_string(m_image->height) + std::string(")")).c_str());
            /** Displayed image color mode. */
        }

        ImGui::Spacing();
    };

private:
    ImageSetInteractor *m_interactor;

    char m_folderPath[256] = {""};

    std::vector<char *> m_comboBoxImagesName;

    size_t m_comboBoxCurrent = 0;

    const Image *m_image = nullptr;
    Texture2D m_imageTex;

    std::string m_loadStatus = std::string("not loaded");

    bool m_enableZoom = true;
};