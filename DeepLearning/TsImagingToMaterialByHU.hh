/* --------------------------------------------------------------------
 * TsImagingToMaterialByHU.hh
 *
 * TOPAS 4.x – Imaging-to-Material extension
 *
 * Purpose
 * -------
 * Maps a CT Hounsfield Unit (HU) value to a material defined by
 * a mass-density and an elemental composition that is read from a
 * user-supplied text file (see example below). The mapping is
 * performed in the master thread during geometry construction and
 * the resulting materials are then used by all worker threads.
 *
 * File format (example – one line per HU entry)
 * --------------------------------------------
 * <HU> <density> <f_H> <f_C> <f_O> ...
 *
 * The composition vector contains the *mass-fractions* of the
 * elements listed in the macro parameter ElementSymbols.
 *
 * Macro parameters
 * ----------------
 * InterpolationFileName   – full path to the text file
 * ElementSymbols          – space-separated list of element symbols
 *
 * Example macro snippet
 * ---------------------
 * setParameter   InterpolationFileName   "InterpolationData.txt"
 * setParameter   ElementSymbols          4 "H" "C" "N" "O"
 *
 * The class inherits from TsVImagingToMaterial, so it can be used
 * exactly like the other imaging-to-material extensions in a
 * geometry component.
 *
 * Thread safety
 * -------------
 * All data structures are fully initialised in the master thread
 * (PreloadAllMaterials()) and are never modified thereafter.
 * Consequently every worker thread may call AssignMaterial()
 * concurrently without any locking.
 *
 * -------------------------------------------------------------------- */

#ifndef TsImagingToMaterialByHU_hh
#define TsImagingToMaterialByHU_hh

// --------------------------------------------------------------------
// TOPAS headers
// --------------------------------------------------------------------
#include "TsVImagingToMaterial.hh"
#include "G4Material.hh"

// --------------------------------------------------------------------
// Standard C++ headers
// --------------------------------------------------------------------
#include <map>
#include <vector>

class TsImagingToMaterialByHU : public TsVImagingToMaterial
{
public:
    TsImagingToMaterialByHU(TsParameterManager* pM, 
            TsVGeometryComponent* component, std::vector<G4Material*>* materialList);

    ~TsImagingToMaterialByHU();

    unsigned short AssignMaterial(std::vector<signed short>* imagingValues, G4int timeSliceIndex);

    void PreloadAllMaterials();

private:  
    void LoadInterpolationTable();

private:
    struct HUProperties {
        G4double density;
        std::vector<G4double> massFractions;
    };
    std::map<G4int, HUProperties> fHUData;
    std::map<G4int, unsigned short> fHUIndexMap;
    std::vector<G4String> fElementSymbols;
};

#endif  // TsImagingToMaterialByHU_hh