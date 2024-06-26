function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function FindAllImageUrls() {
  return Array.from(document.getElementsByTagName("img"))
    .filter(
      (tag) =>
        tag.className == "x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3"
    )
    .map((tag) => tag.src);
}

function GetChildList(element) {
  const list = [element];
  if (!element.children) return list;

  for (const child of Array.from(element.children)) {
    list.push(...GetChildList(child));
  }

  return list;
}

function GetChildrenFromNodeList(list) {
  list = Array.from(list);
  return list.flatMap((node) => GetChildList(node));
}

function WriteToFileAndDownload(filename, text) {
  const element = document.createElement("a");
  element.setAttribute(
    "href",
    "data:text/plain;charset=utf-8," + encodeURIComponent(text)
  );
  element.setAttribute("download", filename);

  element.style.display = "none";
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}

function GetProfileName() {
  const urlSplit = window.location.href.split("/");
  const nameWithQuery = urlSplit[urlSplit.length - 1];
  const queryIdx = nameWithQuery.indexOf("?");
  return queryIdx == -1 ? nameWithQuery : nameWithQuery.slice(0, queryIdx);
}

async function ScrollAndScrape(scrollInterval, scrollDelta, timeLimit = null) {
  const artStyle = prompt("Artstyle?").toLowerCase().trim();
  const imageUrlSet = new Set(FindAllImageUrls());

  const observer = new MutationObserver((mutations) => {
    mutations
      .filter((m) => m.type == "childList")
      .flatMap((m) => m.addedNodes)
      .flatMap((m) => GetChildrenFromNodeList(m))
      .filter((m) => m instanceof HTMLImageElement)
      .map((m) => m.src)
      .forEach((url) => imageUrlSet.add(url));
  });
  const config = { childList: true, subtree: true };
  observer.observe(document.body, config);

  const startTime = new Date().getTime();
  while (true) {
    const heightBefore = window.scrollY;
    window.scrollBy(0, scrollDelta);
    const heightAfter = window.scrollY;

    console.log(heightBefore, heightAfter);

    if (heightAfter - heightBefore <= 1) break;

    const currentTime = new Date().getTime();
    const elapsedTime = (currentTime - startTime) / 1000;
    if (timeLimit != null && elapsedTime > timeLimit) break;

    await sleep(scrollInterval);
  }

  const list = Array.from(imageUrlSet);
  const fileContent = JSON.stringify(
    { entries: list, size: list.length, artStyle },
    null,
    "\t"
  );
  const profileName = GetProfileName();

  WriteToFileAndDownload(profileName, fileContent);
  observer.disconnect();
}

ScrollAndScrape(1500, 500, 10);
