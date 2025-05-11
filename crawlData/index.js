import browser from './config/browser.config.js';
import crawlLinkService from './services/crawlLink.service.js';
import crawlmp3Service from './services/crawlmp3.service.js';
import { writeJson } from './services/fileHandle.service.js';

const main = async () => {
  const page = await browser.newPage();

  // const link = 'https://www.nhaccuatui.com/bai-hat/giu-em-that-lau-naod.TPDciU83yMeU.html';
  // crawlmp3Service(page, link)

  const linkPage = 'https://www.nhaccuatui.com/bai-hat/pop-moi.html';
  const tagLink = "a.avatar_song";
  const links = await crawlLinkService(page, linkPage,tagLink);
  await writeJson('./LinkCrawled/pop.json', links);
  


  await browser.close();


};

main().catch((error) => {
  console.error('Error:', error);
});
