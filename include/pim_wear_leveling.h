#ifndef PIMTORCH_PIM_WEAR_LEVELING_H
#define PIMTORCH_PIM_WEAR_LEVELING_H

class PimWearLevelingManager {
public:
  PimWearLevelingManager(std::vector<PimArrayPtr> pimArrayVec, bool print_detail = true) {
    this->pimArrayVec = pimArrayVec;
    this->print_detail = print_detail;

    // wear leveling
    auto initMap = [&](pimArrayPro* wb_t_ptr) {
      for (int j = 0; j < wb_t_ptr->arr.size(); j++) {
        for (int k = 0; k < wb_t_ptr->arr[j].size(); k++) {
          phyArr2idx.insert(std::make_pair(&pimArrayPro::phyArrManPro[wb_t_ptr->arr[j][k]], wb_t_ptr->arr[j][k]));
          idx2pimArr.insert(std::make_pair(wb_t_ptr->arr[j][k], wb_t_ptr));
          phyArrLastWear.insert(std::make_pair(&pimArrayPro::phyArrManPro[wb_t_ptr->arr[j][k]], 0));
        }
      }
      for (int j = 0; j < wb_t_ptr->narr.size(); j++) {
        for (int k = 0; k < wb_t_ptr->narr[j].size(); k++) {
          phyArr2idx.insert(std::make_pair(&pimArrayPro::phyArrManPro[wb_t_ptr->narr[j][k]], wb_t_ptr->narr[j][k]));
          idx2pimArr.insert(std::make_pair(wb_t_ptr->narr[j][k], wb_t_ptr));
          phyArrLastWear.insert(std::make_pair(&pimArrayPro::phyArrManPro[wb_t_ptr->narr[j][k]], 0));
        }
      }
    };
    for (auto &it : pimArrayVec) {
      initMap(dynamic_cast<pimArrayPro*>(it.ptr.get()));
    }
  }

  void print_wear_info() {
    int64_t sum = 0, x;
    double mean = 0, err = 0;
    std::vector<std::pair<phyArrayPro *, int64_t>> phyArrayVec;

    for (auto &it: phyArr2idx) {
      x = it.first->totalCmpWrCnt;
      phyArrayVec.push_back(std::make_pair(it.first, x));
      sum += x;
    }
    mean = (double)sum / phyArr2idx.size();
    for (auto &it: phyArr2idx) {
      x = it.first->totalCmpWrCnt;
      err += (x - mean) * (x - mean);
    }
    std::sort(phyArrayVec.begin(), phyArrayVec.end(),
              [=](std::pair<phyArrayPro *, int64_t> &a, std::pair<phyArrayPro *, int64_t> &b) {
                return a.first->totalCmpWrCnt < b.first->totalCmpWrCnt;
              });

    std::cout << "variance of phy array wear count: " << err / phyArr2idx.size() << std::endl;
  }

  void rebuild_idx2pimArr(PimArrayPtr &p) {
    pimArrayPro* pimArr = dynamic_cast<pimArrayPro*>(p.ptr.get());
    for (int j = 0; j < pimArr->arr.size(); j++) {
      for (int k = 0; k < pimArr->arr[j].size(); k++) {
        idx2pimArr.insert(std::make_pair(pimArr->arr[j][k], pimArr));
      }
    }
    for (int j = 0; j < pimArr->narr.size(); j++) {
      for (int k = 0; k < pimArr->narr[j].size(); k++) {
        idx2pimArr.insert(std::make_pair(pimArr->narr[j][k], pimArr));
      }
    }
  }

  void wear_leveling() {
    // 1. order phy array wear count
    // 2. order phy array wear count in last interval window
    // 3. map phy array to new pim array

    typedef std::vector<std::vector<int>> Int2DVec;

    // create array vector
    std::vector<std::pair<phyArrayPro *, int64_t>> phyWear;
    std::vector<std::pair<phyArrayPro *, int64_t>> phyIntervalWear;
    std::map<pimArrayPro *, Int2DVec> pim2arrIdxVec, pim2narrIdxVec;
    int64_t lastWearCount;

    auto init_idx_vec = [&](PimArrayPtr &p) {
      pimArrayPro* pimArr = dynamic_cast<pimArrayPro*>(p.ptr.get());
      pim2arrIdxVec.insert(std::make_pair(pimArr, Int2DVec(pimArr->arr)));
      pim2narrIdxVec.insert(std::make_pair(pimArr, Int2DVec(pimArr->narr)));
    };

    auto set_idx_vec = [&](PimArrayPtr &p) {
      pimArrayPro* pimArr = dynamic_cast<pimArrayPro*>(p.ptr.get());
      pimArr->arr = pim2arrIdxVec.find(pimArr)->second;
      pimArr->narr = pim2narrIdxVec.find(pimArr)->second;
    };

    auto erase_phy = [&](std::vector<std::pair<phyArrayPro *, int64_t>> &vec, phyArrayPro *phyArr) {
      for (auto it = vec.begin(); it != vec.end(); it++) {
        if (it->first == phyArr) {
          vec.erase(it);
          break;
        }
      }
    };

    for (auto &it : pimArrayVec) {
      init_idx_vec(it);
    }

    for (auto &it: phyArrLastWear) {
      lastWearCount = phyArrLastWear.find(it.first)->second;
      phyIntervalWear.emplace_back(std::make_pair(it.first, it.first->totalCmpWrCnt - lastWearCount));
      phyWear.emplace_back(std::make_pair(it.first, it.first->totalCmpWrCnt));
    }

    std::sort(phyWear.begin(), phyWear.end(),
              [=](std::pair<phyArrayPro *, int64_t> &a, std::pair<phyArrayPro *, int64_t> &b) {
                return a.first->totalCmpWrCnt > b.first->totalCmpWrCnt;
              });
    std::sort(phyIntervalWear.begin(), phyIntervalWear.end(),
              [=](std::pair<phyArrayPro *, int64_t> &a, std::pair<phyArrayPro *, int64_t> &b) {
                return a.second < b.second;
              });

//    printf("===== before wear leveling =====\n");
//    for (auto &it : phyIntervalWear) {
//      printf("[%d]: %p, last interval wear: %lld, wear count: %lld\n", phyArr2idx.find(it.first)->second, it.first,
//             it.second, it.first->totalCmpWrCnt);
//    }

    auto change_array_idx = [&](pimArrayPro *old_pim_arr, Int2DVec &new_arr_idx,
                                Int2DVec &new_narr_idx, int old_idx, int new_idx) {
      bool change = false;
      for (int j = 0; j < old_pim_arr->arr.size() && !change; j++) {
        for (int k = 0; k < old_pim_arr->arr[j].size() && !change; k++) {
          if (old_pim_arr->arr[j][k] == old_idx) {
            new_arr_idx[j][k] = new_idx;
            change = true;
          }
        }
      }
      for (int j = 0; j < old_pim_arr->narr.size() && !change; j++) {
        for (int k = 0; k < old_pim_arr->narr[j].size() && !change; k++) {
          if (old_pim_arr->narr[j][k] == old_idx) {
            new_narr_idx[j][k] = new_idx;
            change = true;
          }
        }
      }
    };

    while (!phyWear.empty()) {
      int intervalIdx = phyArr2idx.find(phyIntervalWear.back().first)->second;
      int wearIdx = phyArr2idx.find(phyWear.back().first)->second;
      pimArrayPro *intervalPimPtr = idx2pimArr.find(intervalIdx)->second;
      if (intervalIdx != wearIdx) {
        if (print_detail) {
          printf("pim array: %p, [%p] %d(%lld) <-> [%p] %d(%lld)\n", intervalPimPtr,
               phyIntervalWear.back().first, intervalIdx, phyIntervalWear.back().second,
               phyWear.back().first, wearIdx, phyWear.back().second);
        }

        change_array_idx(intervalPimPtr, pim2arrIdxVec.find(intervalPimPtr)->second,
                         pim2narrIdxVec.find(intervalPimPtr)->second, intervalIdx, wearIdx);
      }
      erase_phy(phyWear, phyWear.back().first);
      erase_phy(phyIntervalWear, phyIntervalWear.back().first);
    }

    idx2pimArr.clear();
    for (auto &it : pimArrayVec) {
      set_idx_vec(it);
      rebuild_idx2pimArr(it);
    }

    // update last wear info
    for (auto &it: phyArrLastWear) {
      it.second = it.first->totalCmpWrCnt;
    }

    if (print_detail) {
      print_wear_info();
    }
  }

private:
  bool print_detail;
  std::vector<PimArrayPtr> pimArrayVec;
  std::map<phyArrayPro*, int> phyArr2idx;
  std::map<int, pimArrayPro*> idx2pimArr;
  std::map<phyArrayPro *, int64_t> phyArrLastWear;
};

#endif //PIMTORCH_PIM_WEAR_LEVELING_H
