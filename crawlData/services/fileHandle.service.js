import fs from "fs-extra";

export const writeJson  = async(filePath, data)=>{
  try {
    if(data.length==0){
      console.log(`data is empty!`)
      return
    }
    await fs.writeJson(filePath, data, { spaces: 2 });
    console.log(`write data to ${filePath} successfully!`)
  } catch (err) {
    console.log(`write data to ${filePath} unsuccessfully!`)
    console.error(err);
  }
}

export const readJson = async(filePath)=>{
  try {
    const packageObj = await fs.readJson(filePath);
    console.log(`read data from ${filePath} successfully!`)
    return packageObj;
  } catch (err) {
    console.log(`read data from ${filePath} unsuccessfully!`)
    console.error(err);
  }
}
