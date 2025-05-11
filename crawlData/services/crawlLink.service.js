export default async (page, link, atagSelector)=>{
  await page.goto(link);

  const allArticles = await page.evaluate((selector) => { 
    let articles = document.querySelectorAll(selector);
    return [...articles].map((article) => {
      return new URL(article.getAttribute("href"), window.location.origin).href; 
    });
  }, atagSelector);

  return allArticles;
}