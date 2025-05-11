import fs from "fs-extra";
import path from "path";
export default async (page, link)=>{
  const downloadPath = path.resolve('./DataCrawled');
  fs.mkdirSync(downloadPath, { recursive: true });

  const client = await page.target().createCDPSession();
  await client.send('Page.setDownloadBehavior', {
    behavior: 'allow',
    downloadPath: downloadPath,
  });

  await page.goto(link);

  await page.click('a#btnDownloadBox');
  await new Promise(resolve => setTimeout(resolve, 2000));
  await page.click('a#downloadBasic');
  await new Promise(resolve => setTimeout(resolve, 1000));
}