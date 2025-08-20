// Imaging to Material Converter for MyImagingToMaterialConverter1

#include "TsImagingToMaterialByHU.hh"
#include "TsParameterManager.hh"

#include "G4NistManager.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4ios.hh"

#include <fstream>
#include <sstream>
#include <algorithm>

TsImagingToMaterialByHU::TsImagingToMaterialByHU(TsParameterManager* pM, 
    TsVGeometryComponent* component, std::vector<G4Material*>* materialList) : 
TsVImagingToMaterial(pM, component, materialList)
{
    G4int numberOfComponents = fPm->GetVectorLength(GetFullParmName("ElementSymbols"));
    G4cout << G4endl;
    G4cout << "TsImagingToMaterialByHU: Number of elements: " << numberOfComponents << G4endl;

    G4String* symbols = fPm->GetStringVector(GetFullParmName("ElementSymbols"));
    for (G4int i = 0; i < numberOfComponents; ++i) {
        fElementSymbols.push_back(symbols[i]);
    }
    delete[] symbols;

}

// Destructor
TsImagingToMaterialByHU::~TsImagingToMaterialByHU() {
}

void TsImagingToMaterialByHU::PreloadAllMaterials() {
    if(!LoadInterpolationTable())
    {
        G4cerr << "TsImagingToMaterialByHU: Failed to load interpolation table." << G4endl;
        fPm->AbortSession(1);
    }
}

bool TsImagingToMaterialByHU::LoadInterpolationTable() {
   G4String interpolationFile = fPm->GetStringParameter(GetFullParmName("InterpolationData"));
   G4cout << "Loading interpolation table from file: " << interpolationFile << G4endl;

   std::ifstream inputFile(interpolationFile);
   if (!inputFile.is_open()) {
       G4cerr << "TOPAS is exiting due to a serious error." << G4endl;
       G4cerr << "TsImagingToMaterialByHU: Failed to open interpolation data file: " << interpolationFile << G4endl;
       fPm->AbortSession(1);
   }

    std::string line;
    std::vector<std::pair<G4int, HUProperties>> tempVector;
    while (std::getline(inputFile, line)) {
        std::stringstream ss(line);
        G4int hu;
        G4double density;
        std::vector<G4double> elementFractions(fElementSymbols.size());
        
        ss >> hu >> density;

        for (size_t i = 0; i < elementFractions.size(); ++i) {
            ss >> elementFractions[i];
        }

        tempVector.push_back({hu, {density, elementFractions}});
    }
    inputFile.close();

    // Sort the vector by HU value before populating the map.
    // This is a robust way to handle unsorted input files.
    std::sort(tempVector.begin(), tempVector.end(), 
        [](const std::pair<G4int, HUProperties>& a, const std::pair<G4int, HUProperties>& b) {
            return a.first < b.first;
        });

    // Populate the fHUData map from the sorted vector.
    // This provides efficient lookup later.
    for (const auto& pair : tempVector) {
        fHUData[pair.first] = pair.second;
    }
    
    G4cout << "Successfully loaded and sorted " << fHUData.size() << " unique HU data points." << G4endl;
    return true;
}

unsigned short TsImagingToMaterialByHU::AssignMaterial(std::vector<signed short>* imagingValues, G4int)
{
    // The raw HU value from the voxel
    signed short rawHU = (*imagingValues)[0];

    // First, find the closest HU value from the lookup table ---
    auto upper = fHUData.upper_bound(rawHU);
    auto lower = fHUData.begin();

    // Handle edge cases where the HU is outside our data tableRange
    signed short closestHU;
    if (upper == fHUData.begin()) {
        // The HU is lower than the lowest value in the table, use the lowest.
        closestHU = upper->first;
    }
    else if (upper == fHUData.end()) {
        // The HU is higher than the highest value in the table, use the highest.
        closestHU = std::prev(upper)->first;
    }
    else {
        // The HU is between two values in the table. Find the one that's closer.
        lower = std::prev(upper);

        G4int lowerDiff = std::abs(rawHU - lower->first);
        G4int upperDiff = std::abs(upper->first - rawHU);

        if (upperDiff < lowerDiff) {
            closestHU = upper->first;
        } else {
            closestHU = lower->first;
        }
    }

    // Now, check the cache map with the 'closest' HU value ---
    // This prevents creating a new material for every voxel that isn't an exact match.
    auto it = fHUIndexMap.find(closestHU);
    if (it != fHUIndexMap.end()) {
        // A material for this specific 'closest' HU value has already been created.
        // Simply return its index from the map.
        return it->second;
    }

    // --- Only if the material for 'closestHU' does NOT exist, do we create it ---

    // Get properties for the closest HU value
    auto materialDataIter = fHUData.find(closestHU);
    if (materialDataIter == fHUData.end()) {
        // This should not happen if the logic above is correct, but it's a good guard.
        G4cerr << "TOPAS is exiting due to a serious error." << G4endl;
        G4cerr << "TsImagingToMaterialByHU: Could not find material data for closest HU " << closestHU << G4endl;
        fPm->AbortSession(1);
    }
    const auto& materialProperties = materialDataIter->second;

    // Create a unique name for the material based on its closest HU value
    G4String materialName = "HU_material_" + std::to_string(closestHU);
    G4double density = materialProperties.density;
    G4int nFractions = materialProperties.massFractions.size();

    // Create a new material, for simplicity, assuming a default state/temperature/pressure
    G4Material* material = new G4Material(materialName, density, nFractions);

    // Add elements to the new material, we assume elements are in the same order
    G4double totalFraction = 0.0;
    for (G4int i = 0; i < nFractions; ++i) {
        material->AddMaterial(GetMaterial(fElementSymbols[i]), materialProperties.massFractions[i]);
        totalFraction += materialProperties.massFractions[i];
    }

    if (totalFraction < .9999 || totalFraction > 1.0001) {
        G4cerr << "TOPAS is exiting due to a serious error." << G4endl;
        G4cerr << "TsImagingToMaterialByHU: Total mass fraction for material with HU " << closestHU << " is not 1.0." << G4endl;
        fPm->AbortSession(1);
    }

    // Get the index for this new material.
    unsigned short newMaterialIndex = static_cast<unsigned short>(fMaterialList->size());

    // Add the newly created material pointer to our member vector.
    // NOTE: fMaterialList is inherited from the base class TsVImagingToMaterial.
    fMaterialList->push_back(material);

    // Store the HU value and its corresponding material index for quick lookups.
    fHUIndexMap[closestHU] = newMaterialIndex;

    // Return the index of the newly created material.
    return newMaterialIndex;
} 