const puppeteer = require('puppeteer'); 
 
(async () => { 
	// Initiate the browser 
	const browser = await puppeteer.launch(); 
	 
	// Create a new page with the default browser context 
	const page = await browser.newPage(); 
 
	// Go to the target website 
	await page.goto('view-source:https://www.nordstromrack.com/shop/Men/Clothing/T-Shirts?breadcrumb=Home%2FMen%2FClothing%2FT-Shirts&origin=topnav'); 
 
	// Get pages HTML content 
	const content = await page.content(); 
    //get all of the article elements
    const articles = await page.$$('article');

	console.log(content); 
 
	// Closes the browser and all of its pages 
	await browser.close(); 
})();
