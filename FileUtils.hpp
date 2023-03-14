#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>


class FileUtils {
public:
    FileUtils(){}

    static bool ReadFile(const char* filename, std::string& out_file){
        std::ifstream file(filename);
        bool ret = false;
        
        if(file.is_open()){
            std::string line;
            while(getline(file, line)){
                out_file.append(line);
                out_file.append("\n");
            }
            file.close();
            ret = true;
        }else{
            std::cerr << "Error file: " << filename << " cannot be opened." << std::endl;
        }
        return ret;
    }
};

#endif //FILE_UTILS_H