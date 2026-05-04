# **Practice & Homework Guide**

- github 코드 pull 받는 방법
    - link_rl 폴더에서 하위 폴더인 _01_code 변경 사항 초기화
        - git checkout -- _01_code
    - link_rl 폴더에서 link_rl 리포지토리 최신화
        - git pull
- 추후 구체적인 숙제 내용 제시
- 숙제 제출 사이트: [http://lms.koreatech.ac.kr](http://lms.koreatech.ac.kr/)의 숙제게시판 활용
- 숙제 제출 방법 (jupyter notebook 활용)
    - 1) jupyter notebook 환경에서 각 숙제별로 ipynb 파일 생성
        - [jupyter notebook 사용법 -1](http://www.slideshare.net/TaeYoungLee1/20150306-ipython)
        - [jupyter notebook 사용법 -2](https://www.slideshare.net/dahlmoon/jupyter-notebok-20160815)
        - Python 3.10 버전 이상으로 코딩
    - 2) ipynb 파일 내에 코드를 작성하고 작성한 코드 설명시에 Markdown 문법으로 입력해야 함
        - Markdown 사용법은 각자 익힐 것
        - [Markdown 사용법](https://gist.github.com/ihoneymon/652be052a0727ad59601)
        - [Markdown Tutorial](https://guides.github.com/features/mastering-markdown/)
        - [Markdown Online Editor](http://dillinger.io/)
    - 3) 생성한 ipynb 파일에 대해 자신만의 방법을 사용하여 온라인상 URL을 생성
        - 로컬 파일을 원격으로 업로드하고 URL을 얻으려면 Google Drive, Dropbox 또는 OneDrive와 같은 클라우드 스토리지 서비스를 사용할 수 있음.
        - 다음은 Google 드라이브에 로컬 파일을 업로드하고 URL을 얻는 단계임:
            1. Google 드라이브 계정에 로그인합니다.
            2. "새로 만들기" 버튼을 클릭하고 "파일 업로드"를 선택합니다.
            3. 선택 업로드할 로컬 파일.
            4. 파일이 업로드되면 파일을 마우스 오른쪽 버튼으로 클릭하고 "링크 가져오기"를 선택합니다.
            5. 팝업 창에서 " 링크가 있는 사람은 누구나 볼 수 있습니다."
            6. 제공된 링크를 복사하면 업로드된 파일의 URL이 됩니다.
    - 4) 다음 사이트에 해당 URL을 입력
        - [http://nbviewer.jupyter.org](http://nbviewer.jupyter.org/)
        - 위 nbviewer 사이트를 통하여 보여지는 자신의 숙제를 확인하고 해당 nbviewer URL을 EL2의 자유게시판 본문에 글쓰기로 등록
            - 따라서, 숙제 등록 URL은 반드시 http://nbviewer.jupyter.org/ 로 시작해야 함.
            - 숙제 등록 URL 예: http://nbviewer.jupyter.org/urls/dl.dropbox.com/s/t9nmklgjkp7w4ok/kmeans.ipynb
    - 5) jupyter notebook 단축키
        - ESC, a: 현재 셀 바로 위에 새로운 코드 셀 추가
        - ESC, b: 현재 셀 바로 아래에 새로운 코드 셀 추가
        - ESC, dd: 현재 셀 삭제
        - ESC, m: 현재 셀을 markdown 셀로 변환
        - ESC, y: 현재 셀을 code 셀로 변환
        - ESC, c: 현재 셀 복사
        - ESC, y: 현재 셀 잘라내기
        - ESC, v: 복사하거나 잘라낸 셀을 붙여넣기