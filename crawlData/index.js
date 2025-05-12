import browser from './config/browser.config.js';
import crawlLinkService from './services/crawlLink.service.js';
import crawlmp3Service from './services/crawlmp3.service.js';
import { writeJson } from './services/fileHandle.service.js';
import fs from "fs-extra";
import path from "path";
const main = async () => {
  const page = await browser.newPage();

  // let links = []
  // for (let i = 1; i <= 15; i++){
  //   const linkPage = `https://www.nhaccuatui.com/bai-hat/rock-viet-moi.${i}.html`;
  //   const tagLink = "a.avatar_song";
  //   const linkCrawls = await crawlLinkService(page, linkPage,tagLink);
  //   links.push(...linkCrawls);
  // }
  // await writeJson('./LinkCrawled/rockviet.json', links);

  await crawlmp3Service(page, './LinkCrawled/nhactre.json', './DataCrawled/nhactre2')

  // const desPathFolder = './DataCrawled/nhactre';
  // const downloadPath = path.resolve(desPathFolder);
  // fs.mkdirSync(downloadPath, { recursive: true });

  // const client = await page.target().createCDPSession();
  // await client.send('Page.setDownloadBehavior', {
  //   behavior: 'allow',
  //   downloadPath: downloadPath,
  // });

  // await page.goto('https://www.nhaccuatui.com/playlist/top-100-nhac-tre-hay-nhat-various-artists.m3liaiy6vVsF.html');
  // for (let i = 1; i <= 100; i++) {
  //   await page.evaluate((i) => {
  //     const downloadBtn = document.querySelector(`a.button_download[index="${i}"]`);
  //     if (downloadBtn) downloadBtn.click();
  //   }, i);
  //   await new Promise(resolve => setTimeout(resolve, 2000));
  //   await page.evaluate(() => {
  //     const basicDownload = document.querySelector('a#downloadBasic');
  //     if (basicDownload) basicDownload.click();
  //     else {
  //       async function closeAfterDelay() {
  //         await new Promise(resolve => setTimeout(resolve, 1000));
  //         const btnClose = document.querySelector('a.light_close');
  //         if (btnClose) btnClose.click();
  //       }
  //       closeAfterDelay();
  //     }
  //   });
  //   await new Promise(resolve => setTimeout(resolve, 1000));
    // await page.click('a#btnClose');
    // await new Promise(resolve => setTimeout(resolve, 1000));

  // }



  await browser.close();


};

main().catch((error) => {
  console.error('Error:', error);
});
