#pragma once

#include <vector>
#include <algorithm>

#define UNIQ_ID_MAX 32000

class UniqId {

public:
    /**
     * @brief Construct a new Uniq Id List.
     * This class is not static and can be instanciated couple of times.
     * Each instance is independant.
     * 
     * @param size : The initial size of the memory used to store "reserved" ids.
     */
    UniqId(int size) { m_ids = std::vector<int>(size); };

    /**
     * @brief Add an existing id to the list.
     * 
     * @param id
     * 
     * @return True if id have been added. False if not added (because it already exists.) 
     */
    bool AddId(int id){
        if(IdExists(id) == false){
            m_ids.push_back(id);
            return true;
        }
        return false;
    }

    /**
     * @brief Get a uniq-id. The id is saved in this class as "used".
     * 
     * @return int : A uniq-id represented as an integer. Or -1 if there is no available id.  
     */
    int GetUniqId() {
        int tmp_id = 0;

        for(int i=0; i<UNIQ_ID_MAX; ++i){
            if(IdExists(tmp_id) == false){
                AddId(tmp_id);
                return tmp_id;
            }
            tmp_id += 1;
        }

        return -1;
    };

    /**
     * @brief Check if a uniq-id exists or not. 
     * 
     * @param id : The id to check.
     * @return true : If the id exists.
     * @return false : If the id does not exists.
     */
    bool IdExists(int id){
        auto it = std::find(m_ids.begin(), m_ids.end(), id);
        if (it != m_ids.end()) {
            return true;
        }
        return false;
    };

private:
    std::vector<int> m_ids;
};